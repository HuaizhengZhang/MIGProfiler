import os
from pathlib import Path

import numpy as np
import pandas as pd
import logging
import time
import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from utils.data_hub import load_amazaon_review_data
from utils.model_hub import load_nlp_model


@hydra.main(version_base=None, config_path='../configs', config_name='nlp_infer')
def main(cfg: DictConfig):
    logger = logging.getLogger(cfg.model_name + ' infer')
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
    latency_mean, latency_std, throughput, start_timestamp, end_timestamp = nlp_fixed_time_infer(
        model=model, fixed_time=cfg.fixed_time,
        dataloader=dataloader, device=f'cuda:{cfg.gpu}',
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
    result_file = Path(cfg.result_dir) / 'nlp_infer.csv'
    try:
        if result_file.exists():
            result.to_csv(result_file, header=False, mode='a')
        else:
            result.to_csv(result_file, header=True, mode='w')
    except Exception as e:
        logger.error(f'Errors happen when try to write result to file: {result_file}, {e}')
    logger.info(f'infer results:\n{result}')


def nlp_fixed_time_infer(model, fixed_time, dataloader, device):
    model = model.to(device)
    model.eval()
    latency = []
    total_sample = 0
    for i, inputs in tqdm(enumerate(dataloader)):
        inputs = inputs.to(device)
        with torch.set_grad_enabled(True):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(**inputs)
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

