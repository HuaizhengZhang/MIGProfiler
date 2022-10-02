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
import pynvml

from utils.common import p99_latency
from utils.data_hub import load_places365_data
from utils.model_hub import load_cv_model
pynvml.nvmlInit()


@hydra.main(version_base=None, config_path='../configs', config_name='cv_infer')
def main(cfg: DictConfig):
    if not Path(cfg.result_dir).exists():
        os.makedirs(cfg.result_dir)
    # create model
    model, input_size = load_cv_model(model_name=cfg.model_name)
    dataloader = load_places365_data(
            input_size=input_size,
            batch_size=cfg.batch_size,
            data_path=cfg.data_path,
            num_workers=cfg.workers
        )
    tail_latency, latency_std, throughput, power_mean, start_timestamp, end_timestamp = cv_fixed_time_infer(
        model=model, fixed_time=cfg.fixed_time,
        dataloader=dataloader, device=f'cuda:{cfg.gpu}',
    )
    result = pd.DataFrame({
        'model_name': [cfg.model_name],
        'batch_size':[cfg.batch_size],
        'latency': [tail_latency],
        'latency_std': [latency_std],
        'throughput': [throughput],
        'power': [power_mean],
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
        print(f'Errors happen when try to write result to file: {result_file}, {e}')
    print(f'infer results:\n{result}')


def cv_fixed_time_infer(model, fixed_time, dataloader, device):
    model = model.to(device)
    model.eval()
    latency = []
    total_sample = 0
    power_usage = []
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    for i, (inputs, labels) in tqdm(enumerate(dataloader)):
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
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
                power_usage += [pynvml.nvmlDeviceGetPowerUsage(handle)/1000]
        if i > 100 and end_timestamp - start_timestamp > fixed_time:
            break
    throughput = float(1000 * total_sample) / np.sum(latency)
    tail_latency = p99_latency(latency)
    latency_std = np.std(latency)
    power_usage_mean = np.mean(power_usage)
    # gpu clear
    torch.cuda.empty_cache()

    return tail_latency, latency_std, throughput, power_usage_mean, start_timestamp, end_timestamp


if __name__ == "__main__":
    main()
