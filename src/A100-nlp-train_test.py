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

from utils.data_hub import load_amazaon_review_data
from utils.model_hub import load_nlp_model


@hydra.main(version_base=None, config_path='../configs', config_name='nlp_train')
def main(cfg: DictConfig):
    logger = logging.getLogger(cfg.model_name + ' train')
    if not Path(cfg.result_dir).exists():
        os.makedirs(cfg.result_dir)
    # create model
    logger.info("getting model '{}' from torch hub".format(cfg.model_name))
    model = load_nlp_model(cfg.model_name)
    dataloader = load_amazaon_review_data(
        model_name=cfg.model_name,
        seq_length=cfg.seq_length,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers
    )
    logger.info("model: '{}' is successfully loaded".format(model.__class__.__name__))
    criterion = nn.CrossEntropyLoss().cuda(cfg.gpu)
    optimizer = torch.optim.AdamW(model.parameters(), cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    latency_mean, latency_std, throughput, start_timestamp, end_timestamp = nlp_fixed_time_train(
        model=model, fixed_time=cfg.fixed_time,
        dataloader=dataloader, device=f'cuda:{cfg.gpu}',
        criterion=criterion, optimizer=optimizer
    )
    result = pd.DataFrame({
        'model_name': [cfg.model_name],
        'batch_size': [cfg.batch_size],
        'seq_length': [cfg.seq_length],
        'latency': [latency_mean],
        'latency_std': [latency_std],
        'throughput': [throughput],
        'start_timestamp': [start_timestamp],
        'end_timestamp': [end_timestamp],
        'mig_profile': [cfg.mig_profile]
    }).round(2)
    result_file = Path(cfg.result_dir) / 'nlp_train.csv'
    try:
        if result_file.exists():
            result.to_csv(result_file, header=False, mode='a')
        else:
            result.to_csv(result_file, header=True, mode='w')
    except Exception as e:
        logger.error(f'Errors happen when try to write result to file: {result_file}, {e}')
    logger.info(f'infer results:\n{result}')


def nlp_fixed_time_train(model, fixed_time, dataloader, criterion, optimizer, device):
    model = model.to(device)
    model.eval()
    latency = []
    total_sample = 0
    torch.manual_seed(100)
    for i, inputs in tqdm(enumerate(dataloader)):
        inputs = inputs.to(device)
        labels = torch.randint(0, 2, [len(inputs['input_ids'])]).to(device)
        with torch.set_grad_enabled(True):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            output = model(**inputs)
            loss = criterion(output['logits'], labels)
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

