import builtins
import os
import subprocess
import warnings

import numpy as np
import pandas as pd
import logging
import time
import hydra
import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import torch.distributed as dist
import moco.loader
import moco.builder
import torch.multiprocessing as mp
import random
import torch.backends.cudnn as cudnn

pd.set_option('display.max_columns', None)
DCGM_GTOUP_ID = 15
DCGM_INSTANCE_ID = 0


# gpu metric reference: https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-user-guide/feature-overview.html#profiling-metrics


@hydra.main(version_base=None, config_path='configs', config_name='moco_pretrain')
def main(cfg: DictConfig):
    logger = logging.getLogger(cfg.arch + ' pretrain')
    # start dcgm monitoring
    try:
        logger.info("starting dcgm recoder subprocess...")
        p = subprocess.Popen(['python', '/root/infer_test/dcgm_recorder.py'])
        logger.info("dcgm recoder process pid={} is running now.".format(p.pid))

    except Exception as e:
        logger.info("dcgm recorder failed: {}".format(e))
    try:
        if cfg.seed is not None:
            random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')

        if cfg.gpu is not None:
            warnings.warn('You have chosen a specific GPU. This will completely '
                          'disable data parallelism.')

        if cfg.dist_url == "env://" and cfg.world_size == -1:
            cfg.world_size = int(os.environ["WORLD_SIZE"])

        cfg.distributed = cfg.world_size > 1 or cfg.multiprocessing_distributed

        ngpus_per_node = torch.cuda.device_count()
        if cfg.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            cfg.world_size = ngpus_per_node * cfg.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
        else:
            # Simply call main_worker function
            main_worker(cfg.gpu, ngpus_per_node, cfg)

    except Warning as w:
        logger.warning(f"dcgm process{p.pid} closed due of interruption:{w}")
        p.terminate()
    p.terminate()
    logger.info('benchmark testing completed')


def fixed_time_pretrain_benchmark(model, fixed_time, criterion, dataloader, args, optimizer):
    latency = []
    total_sample = 0
    for i, (images, _) in enumerate(dataloader):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ender.record()
        torch.cuda.synchronize()
        if i == 100:
            start_timestamp = int(time.time())
        if i >= 100:
            latency += [starter.elapsed_time(ender)]
            end_timestamp = int(time.time())
            total_sample += len(images[0])
        if i > 100 and end_timestamp - start_timestamp > fixed_time:
            break
    throughput = float(1000 * total_sample) / np.sum(latency)
    latency_mean = np.mean(latency)
    latency_std = np.std(latency)

    return latency_mean, latency_std, throughput, start_timestamp, end_timestamp


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    train_loader = load_data(args.data, args.batch_size, args.workers, args.aug_plus, args.distributed)
    latency_mean, latency_std, throughput, start_timestamp, end_timestamp = \
        fixed_time_pretrain_benchmark(model, 120, criterion, train_loader, args, optimizer)
    print("batch_size:{}, latency_mean:{}, latency_std:{}, throughput:{}, start_timestamp:{}, end_timestamp:{}".format(
        args.batch_size, latency_mean, latency_std, throughput, start_timestamp, end_timestamp))



# one epoch train for test
def train(train_loader, model, criterion, optimizer, args):
    # switch to train mode
    model.train()
    for i, (images, _) in enumerate(train_loader):
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def load_data(data, batch_size, num_workers, aug_plus=True, distributed=True):
    # Data loading code
    traindir = os.path.join(data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    train_dataset = datasets.ImageFolder(
        traindir,
        moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    return train_loader


if __name__ == "__main__":
    main()
