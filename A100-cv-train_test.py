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
from tqdm import tqdm

from utils.data_hub import load_places365_data
from utils.model_hub import load_cv_model
# gpu metric reference: https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-user-guide/feature-overview.html


@hydra.main(version_base=None, config_path='configs', config_name='cv_train')
def main(cfg: DictConfig):
    logger = logging.getLogger(cfg.model_name+' infer')
    if not Path(cfg.result_dir).exists():
        os.makedirs(cfg.result_dir)
    # create model
    logger.info("getting model '{}' from torch hub".format(cfg.model_name))
    model, input_size = load_cv_model(model_name=cfg.model_name)
    dataloader = load_places365_data(
            input_size=input_size,
            batch_size=cfg.batch_size,
            data_path=cfg.data_path,
            num_workers=cfg.workers
        )
    logger.info("model: '{}' is successfully loaded".format(model.__class__.__name__))
    criterion = nn.CrossEntropyLoss().cuda(cfg.gpu)
    optimizer = torch.optim.SGD(model.parameters(), cfg.optimizer.lr,
                                momentum=cfg.optimizer.momentum,
                                weight_decay=cfg.optimizer.weight_decay)
    latency_mean, latency_std, throughput, start_timestamp, end_timestamp = cv_fixed_time_train(
        model=model, fixed_time=cfg.fixed_time,
        dataloader=dataloader, device=f'cuda:{cfg.gpu}',
        criterion=criterion, optimizer=optimizer
    )
    result = pd.DataFrame({
        'model_name': [cfg.model_name],
        'batch_size': [cfg.batch_size],
        'latency': [latency_mean],
        'latency_std': [latency_std],
        'throughput': [throughput],
        'start_timestamp': [start_timestamp],
        'end_timestamp': [end_timestamp],
        'mig_profile': [cfg.mig_profile]
    }).round(2)
    result_file = Path(cfg.result_dir) / 'cv_train.csv'
    try:
        if result_file.exists():
            result.to_csv(result_file, header=False, mode='a')
        else:
            result.to_csv(result_file, header=True, mode='w')
    except Exception as e:
        logger.error(f'Errors happen when try to write result to file: {result_file}, {e}')
    logger.info(f'train results:\n{result}')


def cv_fixed_time_train(model, fixed_time, dataloader, criterion, optimizer, device):
    model = model.to(device)
    model.eval()
    latency = []
    total_sample = 0
    for i, (inputs, labels) in tqdm(enumerate(dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(True):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            output = model(inputs)
            loss = criterion(output, labels)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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


if __name__ == "__main__":
    main()
