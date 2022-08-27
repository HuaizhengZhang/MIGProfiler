import os
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import time
import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from tqdm import tqdm


@hydra.main(version_base=None, config_path='configs', config_name='cv_infer')
def main(cfg: DictConfig):
    logger = logging.getLogger(cfg.model_name+' infer')
    if not Path(cfg.result_dir).exists():
        os.makedirs(cfg.result_dir)
    # create model
    logger.info("getting model '{}' from torch hub".format(cfg.model_name))
    model, input_size = initialize_model(architecture=cfg.model_name)
    dataloader = load_places365_data(
            input_size=input_size,
            batch_size=cfg.batch_size,
            data_path=cfg.data_path,
            num_workers=cfg.workers
        )
    logger.info("model: '{}' is successfully loaded".format(model.__class__.__name__))
    latency_mean, latency_std, throughput, start_timestamp, end_timestamp = cv_fixed_time_infer(
        model=model, fixed_time=cfg.fixed_time,
        dataloader=dataloader, device=f'cuda:{cfg.gpu}',
    )
    result = pd.DataFrame({
        'model_name': [cfg.model_name],
        'batch_size':[cfg.batch_size],
        'latency': [latency_mean],
        'latency_std': [latency_std],
        'throughput': [throughput],
        'start_timestamp': [start_timestamp],
        'end_timestamp': [end_timestamp],
        'mig_profile': [cfg.mig_profile]
    }).round(2)
    result_file = Path(cfg.result_dir) / 'cv_infer.csv'
    try:
        if result_file.exists():
            result.to_csv(result_file, header=False, mode='a')
        else:
            result.to_csv(result_file, header=True, mode='w')
    except Exception as e:
        logger.error(f'Errors happen when try to write result to file: {result_file}, {e}')
    logger.info(f'infer results:\n{result}')


def cv_fixed_time_infer(model, fixed_time, dataloader, device):
    model = model.to(device)
    model.eval()
    latency = []
    total_sample = 0
    for i, (inputs, labels) in tqdm(enumerate(dataloader)):
        inputs = inputs.to(device)
        with torch.set_grad_enabled(True):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(inputs)
            ender.record()
            torch.cuda.synchronize()
            if i == 20:
                start_timestamp = int(time.time())
            if i >= 20:
                latency += [starter.elapsed_time(ender)]
                end_timestamp = int(time.time())
                total_sample += len(inputs)
        if i > 100 and end_timestamp - start_timestamp > fixed_time:
            break
    throughput = float(1000 * total_sample) / np.sum(latency)
    latency_mean = np.mean(latency)
    latency_std = np.std(latency)
    # gpu clear
    torch.cuda.empty_cache()

    return latency_mean, latency_std, throughput, start_timestamp, end_timestamp


def initialize_model(architecture) -> (nn.Module, int):
    """get models from https://pytorch.org/hub/.

    Args:
        architecture: model architecture.
    Return:
        model, input size.
    """
    # Initialize these variables which will be set in this if statement. Each of these variables is model specific.
    """ Vision Transformer base 16
    """
    if architecture == "vision_transformer":
        model_ft = models.vit_b_16(weights='ViT_B_16_Weights.DEFAULT')
        input_size = 224
    elif architecture == "resnet50":
        model_ft = models.resnet50(weights="IMAGENET1K_V1")
        input_size = 224
    elif architecture == "swin_transformer":
        model_ft = models.swin_transformer(pretrained=True)
        input_size = 224
    else:
        raise NotImplementedError(f"{architecture} is not supported, try 'vision_transformer' or 'resnet50', or 'swin_transformer'.")
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

    # Create traininG dataset
    image_dataset = datasets.ImageFolder(data_path, data_transform)
    # Create training and validation dataloaders
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers)
    return dataloader


if __name__ == "__main__":
    main()
