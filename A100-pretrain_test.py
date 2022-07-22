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
        # create model
        print("=> creating model '{}'".format(cfg.arch))
        model = moco.builder.MoCo(
            models.__dict__[cfg.arch],
            cfg.moco_dim, cfg.moco_k, cfg.moco_m, cfg.moco_t, cfg.mlp)
        print(model)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda(cfg.gpu)
        optimizer = torch.optim.SGD(model.parameters(), cfg.lr,
                                    momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
        train_loader = load_data(cfg.data, cfg.batch_size, cfg.workers, cfg.aug_plus)
        latency_mean, latency_std, throughput, start_timestamp, end_timestamp = \
            fixed_time_pretrain_benchmark(model, 120, criterion, train_loader, cfg, optimizer)
        print(
            "batch_size:{}, latency_mean:{}, latency_std:{}, throughput:{}, start_timestamp:{}, end_timestamp:{}".format(
                cfg.batch_size, latency_mean, latency_std, throughput, start_timestamp, end_timestamp))

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


def load_data(data, batch_size, num_workers, aug_plus=True):
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    return train_loader


if __name__ == "__main__":
    main()
