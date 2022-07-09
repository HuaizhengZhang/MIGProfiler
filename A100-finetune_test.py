import os
import subprocess
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
from tqdm import tqdm
from urllib import request
pd.set_option('display.max_columns', None)
DCGM_GTOUP_ID = 15
DCGM_INSTANCE_ID = 0


# gpu metric reference: https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-user-guide/feature-overview.html#profiling-metrics


@hydra.main(version_base=None, config_path='configs', config_name='finetune')
def main(cfg: DictConfig):
    logger = logging.getLogger(cfg.arch+' finetune')

    # create model
    logger.info("getting model '{}' from torch hub".format(cfg.arch))
    model, input_size = initialize_model(
        architecture=cfg.arch,
        num_classes=cfg.num_classes,
        feature_extract=cfg.feature_extract,
    )
    logger.info("model: '{}' is successfully loaded".format(model.__class__.__name__))
    logger.info("model structure: {}".format(model))
    # start dcgm monitoring
    try:
        logger.info("starting dcgm recoder subprocess...")
        p = subprocess.Popen(['python', '/root/infer_test/dcgm_recorder.py'])
        logger.info("dcgm recoder process pid={} is running now.".format(p.pid))

    except Exception as e:
        logger.info("dcgm recorder failed: {}".format(e))

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model.parameters()
    param_log_info = ''
    if cfg.feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                param_log_info += "\t{}".format(name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_log_info += "\t{}".format(name)
    logger.info("Params to learn:\n" + param_log_info)
    # Detect if we have a GPU available
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=cfg.lr, momentum=cfg.momentum)
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # benchmark
    easy_finetune_experiment(model=model, input_size=input_size, result_dir='./',
                             data_path=cfg.data_path, batch_sizes=cfg.batch_sizes,
                             device=device, criterion=criterion, optimizer_ft=optimizer_ft,
                             logger=logger, fixed_time=True)
    p.terminate()


def easy_finetune_experiment(
        model,
        input_size,
        result_dir,
        data_path,
        batch_sizes,
        device,
        criterion,
        optimizer_ft,
        logger,
        workers=6,
        fixed_time: bool = False):
    latency_mean_list = []
    latency_std_list = []
    throughput_list = []
    start_timestamp_list = []
    end_timestamp_list = []
    for batch_size in batch_sizes:
        # gpu clear and cool down
        torch.cuda.empty_cache()
        model = model.to('cpu')
        time.sleep(15)
        logger.info("Initializing Datasets and Dataloaders...")
        logger.info("loading training data from {}".format(data_path))
        dataloader = load_places365_data(
            input_size=input_size,
            batch_size=batch_size,
            data_path=data_path,
            num_workers=workers
        )
        if fixed_time:
            latency_mean, latency_std, throughput, start_timestamp, end_timestamp = fixed_time_finetune_benchmark(
                model=model,
                dataloader=dataloader,
                device=device,
                criterion=criterion,
                optimizer=optimizer_ft,
                logger=logger,
                fixed_time=120,
            )
        else:
            latency_mean, latency_std, throughput, start_timestamp, end_timestamp = fixed_iter_finetune_benchmark(
                model=model,
                iteration=1000,
                dataloader=dataloader,
                device=device,
                criterion=criterion,
                optimizer=optimizer_ft,
                logger=logger,
            )
        # logger.info("subprocess {} of dcgm recoder is terminated".format(p.pid))
        logger.info(
            "batch size:{}, latency_mean:{}, leantency_std:{}, throughput:{}, start_timestamp:{}, end_timestamp:{}".format(
                batch_size, latency_mean, latency_std, throughput, start_timestamp, end_timestamp))
        latency_mean_list += [latency_mean]
        latency_std_list += [latency_std]
        throughput_list += [throughput]
        start_timestamp_list += [start_timestamp]
        end_timestamp_list += [end_timestamp]
        del dataloader
    result = pd.DataFrame({
        'model_name': ['resnet50_moco'] * len(batch_sizes),
        'batch_size': batch_sizes,
        'latency': latency_mean_list,
        'latency_std': latency_std_list,
        'throughput': throughput_list,
        'start_timestamp': start_timestamp_list,
        'end_timestamp': end_timestamp_list}).round(2)
    print(result)
    save_file = result_dir + f'resnet50_moco_40g_bsz.csv'
    keep_head = False if os.path.exists(save_file) else True
    result.to_csv(save_file, mode='a', index=False, header=keep_head)


def fixed_iter_finetune_benchmark(model, iteration, dataloader, criterion, device, optimizer, logger):
    # Send the model to GPU
    model = model.to(device)
    model.train()
    latency = []
    total_sample = 0
    for i, (inputs, labels) in tqdm(enumerate(dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            ender.record()
            torch.cuda.synchronize()
            if i == 100:
                start_timestamp = int(time.time())
            if i >= 100:
                latency += [starter.elapsed_time(ender)]
                end_timestamp = int(time.time())
                total_sample += len(inputs)
        if i >= 100 + iteration:
            break
    assert end_timestamp, "iteration is too large, try to set a smaller one"
    throughput = float(1000 * total_sample) / np.sum(latency)
    logger.info('benchmark testing completed')
    latency_mean = np.mean(latency)
    latency_std = np.std(latency)
    model, inputs, outputs, loss, labels = model.to('cpu'), inputs.to('cpu'), outputs.to('cpu'), loss.to(
        'cpu'), labels.to('cpu')
    del model, inputs, outputs, loss, labels
    # gpu clear and cool down
    torch.cuda.empty_cache()
    time.sleep(10)
    return latency_mean, latency_std, throughput, start_timestamp, end_timestamp


def fixed_time_finetune_benchmark(model, fixed_time, dataloader, criterion, device, optimizer, logger):
    # Send the model to GPU
    model = model.to(device)
    model.train()
    latency = []
    total_sample = 0
    for i, (inputs, labels) in tqdm(enumerate(dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            ender.record()
            torch.cuda.synchronize()
            if i == 100:
                start_timestamp = int(time.time())
            if i >= 100:
                latency += [starter.elapsed_time(ender)]
                end_timestamp = int(time.time())
                total_sample += len(inputs)
        if i > 100 and end_timestamp - start_timestamp > fixed_time:
            break
    assert end_timestamp, "iteration is too large, try to set a smaller one"
    throughput = float(1000 * total_sample) / np.sum(latency)
    logger.info('benchmark testing completed')
    latency_mean = np.mean(latency)
    latency_std = np.std(latency)
    model, inputs, outputs, loss, labels = model.to('cpu'), inputs.to('cpu'), outputs.to('cpu'), loss.to(
        'cpu'), labels.to('cpu')
    del model, inputs, outputs, loss, labels
    # gpu clear and cool down
    torch.cuda.empty_cache()
    time.sleep(10)

    return latency_mean, latency_std, throughput, start_timestamp, end_timestamp


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(architecture, num_classes, feature_extract) -> (nn.Module, int):
    """get models from https://pytorch.org/hub/.

    Args:
        architecture: model architecture.
        num_classes (int): the output dimension of model classifier.
        feature_extract (bool): if true, will freeze all the gradients.
    Return:
        model, input size.
    """
    # Initialize these variables which will be set in this if statement. Each of these variables is model specific.
    """ Vision Transformer base 16
    """
    if architecture == "vision_transformer":
        model_ft = models.vit_b_16(pretrained=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the primary net
        num_ftrs = model_ft.hidden_dim
        model_ft.heads = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif architecture == "resnet50":
        model_ft = models.resnet50(pretrained=False)
        if not os.path.exists("/root/.cache/moco_v2_800ep_pretrain.pth.tar"):
            weights_url = "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar"
            request.urlretrieve(weights_url, "/root/.cache/moco_v2_800ep_pretrain.pth.tar")
        state_dict = torch.load("/root/.cache/moco_v2_800ep_pretrain.pth.tar")
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                k_no_prefix = k[len("module."):]
                state_dict[k_no_prefix] = state_dict[k]  # leave encoder_q in the param name
                # copy state from the query encoder into a new parameter for the key encoder
                state_dict[k_no_prefix.replace('encoder_q', 'encoder_k')] = state_dict[k]
            del state_dict[k]
        model_ft.load_state_dict(state_dict, strict=False)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the primary net
        model_ft.fc = nn.Linear(2048, num_classes)
        input_size = 224
    else:
        raise NotImplementedError(f"{architecture} is not supported, try 'vision_transformer' or 'resnet50'.")
    return model_ft, input_size


def load_places365_data(input_size, data_path, batch_size, num_workers) -> DataLoader:
    """transform data and load data into dataloader. Images should be arranged in this way by default: ::

        root/my_dataset/dog/xxx.png
        root/my_dataset/dog/xxy.png
        root/my_dataset/dog/[...]/xxz.png

        root/my_dataset/cat/123.png
        root/my_dataset/cat/nsdf3.png
        root/my_dataset/cat/[...]/asd932_.png

    Args:
        input_size (int): transformed image resolution, such as 224.
        data_path (string): eg. root/my_dataset/
        batch_size (int): batch size
        num_workers (int): number of pytorch DataLoader worker subprocess
    """
    data_transform = transforms.Compose([
        transforms.Resize(
            input_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create training and validation datasets
    image_dataset = datasets.ImageFolder(data_path, data_transform)
    # Create training and validation dataloaders
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers)
    return dataloader


if __name__ == "__main__":
    main()
