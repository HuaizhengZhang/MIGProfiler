import re
import numpy as np
import pandas as pd
import time
import requests
import torch
from torch import nn
from tqdm import tqdm
from mig_perf.utils.data_hub import load_amazaon_review_data
from mig_perf.utils.model_hub import load_nlp_model
from mig_perf.utils.common import p99_latency
from mig_perf.utils.data_hub import load_places365_data
from mig_perf.utils.model_hub import load_cv_model
# gpu metric reference: https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-user-guide/feature-overview.html

DCGM_URL = "http://dcgm_exporter:9400/metrics"
GPU_I_ID = 1


def cv_infer(
        model_name,
        fixed_time,
        batch_size,
        data_path,
):
    # create model
    model, input_size = load_cv_model(model_name=model_name)
    dataloader = load_places365_data(
            input_size=input_size,
            data_path=data_path,
            batch_size=batch_size,
        )
    tail_latency, latency_std, throughput, gract, gract_std, fbusd, power, profile = _cv_fixed_time_infer(
        model=model, fixed_time=fixed_time,
        dataloader=dataloader, device=f'cuda:0',
        gpu_i_id=GPU_I_ID, dcgm_url=DCGM_URL
    )
    result = pd.DataFrame({
        'model_name': [model_name],
        'batch_size': [batch_size],
        'latency': [tail_latency],
        'latency_std': [latency_std],
        'throughput': [throughput],
        'gract': [float(gract)],
        'gract_std': [gract_std],
        'fbusd': [float(fbusd)],
        'power': [float(power)],
        'mig_profile': [profile]
    }).round(2)
    return result


def _cv_fixed_time_infer(model, gpu_i_id, dcgm_url, fixed_time, dataloader, device):
    model = model.to(device)
    model.eval()
    latency = []
    gract_list = []
    fbusd_list = []
    power_list = []
    total_sample = 0
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
            if i % 10 == 0:
                gract, fbusd, power, profile = dcgm_exporter(gpu_i_id, dcgm_url)
                gract_list.append(float(gract))
                fbusd_list.append(float(fbusd))
                power_list.append(float(power))
        if i > 100 and end_timestamp - start_timestamp > fixed_time:
            break
    throughput = float(1000 * total_sample) / np.sum(latency)
    tail_latency = p99_latency(latency)
    latency_std = np.std(latency)
    gract_mean = np.mean(gract_list)
    gract_std = np.std(gract_list)
    fbusd_mean = np.mean(fbusd_list)
    power_mean = np.mean(power_list)
    # gpu clear
    torch.cuda.empty_cache()

    return tail_latency, latency_std, throughput, gract_mean, gract_std, fbusd_mean, power_mean, profile


def cv_train(
        model_name,
        fixed_time,
        batch_size,
        data_path,
        lr=1e-5,
        momentum=0.9,
        weight_decay=1e-4
):
    # TODO: optimizer should be customized
    # create model
    model, input_size = load_cv_model(model_name=model_name)
    dataloader = load_places365_data(
            input_size=input_size,
            batch_size=batch_size,
            data_path=data_path,
        )
    criterion = nn.CrossEntropyLoss().cuda(0)
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    tail_latency, latency_std, throughput, gract, gract_std, fbusd, power, profile = _cv_fixed_time_train(
        criterion=criterion, optimizer=optimizer,
        model=model, fixed_time=fixed_time,
        dataloader=dataloader, device=f'cuda:0',
        gpu_i_id=GPU_I_ID, dcgm_url=DCGM_URL
    )
    result = pd.DataFrame({
        'model_name': [model_name],
        'batch_size': [batch_size],
        'latency': [tail_latency],
        'latency_std': [latency_std],
        'throughput': [throughput],
        'gract': [float(gract)],
        'gract_std': [gract_std],
        'fbusd': [float(fbusd)],
        'power': [float(power)],
        'mig_profile': [profile]
    }).round(2)
    return result


def _cv_fixed_time_train(model, fixed_time, dataloader, criterion, optimizer, device, gpu_i_id, dcgm_url):
    model = model.to(device)
    model.eval()
    latency = []
    gract_list = []
    fbusd_list = []
    power_list = []
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
            if i % 10 == 0:
                gract, fbusd, power, profile = dcgm_exporter(gpu_i_id, dcgm_url)
                gract_list.append(float(gract))
                fbusd_list.append(float(fbusd))
                power_list.append(float(power))
        if i > 100 and end_timestamp - start_timestamp > fixed_time:
            break
    throughput = float(1000 * total_sample) / np.sum(latency)
    latency_mean = np.mean(latency)
    latency_std = np.std(latency)
    gract_mean = np.mean(gract_list)
    gract_std = np.std(gract_list)
    fbusd_mean = np.mean(fbusd_list)
    power_mean = np.mean(power_list)
    # gpu clear
    torch.cuda.empty_cache()

    return latency_mean, latency_std, throughput, gract_mean, gract_std, fbusd_mean, power_mean, profile


def nlp_infer(
        model_name,
        fixed_time,
        seq_length,
        batch_size,
):
    # create model
    model = load_nlp_model(model_name)
    dataloader = load_amazaon_review_data(
        model_name=model_name,
        seq_length=seq_length,
        batch_size=batch_size,
    )
    latency_mean, latency_std, throughput, gract, gract_std, fbusd, power, profile = _nlp_fixed_time_infer(
        model=model, fixed_time=fixed_time,
        dataloader=dataloader, device=f'cuda:0',
        gpu_i_id=GPU_I_ID,
        dcgm_url=DCGM_URL
    )
    result = pd.DataFrame({
        'model_name': [model_name],
        'batch_size': [batch_size],
        'seq_length': [seq_length],
        'latency': [latency_mean],
        'latency_std': [latency_std],
        'throughput': [throughput],
        'gract': [float(gract)],
        'gract_std': [gract_std],
        'fbusd': [float(fbusd)],
        'power': [float(power)],
        'mig_profile': [profile]
    }).round(2)
    return result


def _nlp_fixed_time_infer(model, fixed_time, dataloader, device, gpu_i_id, dcgm_url):
    model = model.to(device)
    model.eval()
    latency = []
    gract_list = []
    fbusd_list = []
    power_list = []
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
            if i % 10 == 0:
                gract, fbusd, power, profile = dcgm_exporter(gpu_i_id, dcgm_url)
                gract_list.append(float(gract))
                fbusd_list.append(float(fbusd))
                power_list.append(float(power))
        if i > 100 and end_timestamp - start_timestamp > fixed_time:
            break
    throughput = float(1000 * total_sample) / np.sum(latency)
    latency_mean = np.mean(latency)
    latency_std = np.std(latency)
    gract_mean = np.mean(gract_list)
    gract_std = np.std(gract_list)
    fbusd_mean = np.mean(fbusd_list)
    power_mean = np.mean(power_list)
    # gpu clear
    torch.cuda.empty_cache()

    return latency_mean, latency_std, throughput, gract_mean, gract_std, fbusd_mean, power_mean, profile


def nlp_train(
        model_name,
        fixed_time,
        seq_length,
        batch_size,
        lr=1e-5,
        weight_decay=1e-4
):
    # TODO: optimizer should be customized
    # create model
    model = load_nlp_model(model_name)
    dataloader = load_amazaon_review_data(
        model_name=model_name,
        seq_length=seq_length,
        batch_size=batch_size,
    )
    criterion = nn.CrossEntropyLoss().cuda(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
    latency_mean, latency_std, throughput, gract, gract_std, fbusd, power, profile = _nlp_fixed_time_train(
        model=model, fixed_time=fixed_time,
        dataloader=dataloader, device=f'cuda:0',
        criterion=criterion, optimizer=optimizer,
        gpu_i_id=GPU_I_ID, dcgm_url=DCGM_URL
    )
    result = pd.DataFrame({
        'model_name': [model_name],
        'batch_size': [batch_size],
        'seq_length': [seq_length],
        'latency': [latency_mean],
        'latency_std': [latency_std],
        'throughput': [throughput],
        'gract': [float(gract)],
        'gract_std': gract_std,
        'fbusd': [float(fbusd)],
        'power': [float(power)],
        'mig_profile': [profile]
    }).round(2)
    return result


def _nlp_fixed_time_train(model, fixed_time, dataloader, criterion, optimizer, device, gpu_i_id, dcgm_url):
    model = model.to(device)
    model.eval()
    latency = []
    gract_list = []
    fbusd_list = []
    power_list = []
    total_sample = 0
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
            if i % 10 == 0:
                gract, fbusd, power, profile = dcgm_exporter(gpu_i_id, dcgm_url)
                gract_list.append(float(gract))
                fbusd_list.append(float(fbusd))
                power_list.append(float(power))
        if i > 100 and end_timestamp - start_timestamp > fixed_time:
            break
    throughput = float(1000 * total_sample) / np.sum(latency)
    latency_mean = np.mean(latency)
    latency_std = np.std(latency)
    gract_mean = np.mean(gract_list)
    gract_std = np.std(gract_list)
    fbusd_mean = np.mean(fbusd_list)
    power_mean = np.mean(power_list)
    # gpu clear
    torch.cuda.empty_cache()

    return latency_mean, latency_std, throughput, gract_mean, gract_std, fbusd_mean, power_mean, profile


def dcgm_exporter(gpu_i_id, url):
    metric = requests.get(url).text
    if metric is None:
        return None
    for line in metric.splitlines():
        if line[0] != '#':
            gid = re.search(r"GPU_I_ID=\"(.)\"", line).group(1)
            if gid == str(gpu_i_id):
                profile = re.search(r"GPU_I_PROFILE=\"([0-9]g.[0-9]*gb)", line).group(1)
                if line.split('{')[0] == "DCGM_FI_DEV_FB_USED":
                    fbusd = line.split(' ')[-1]
                if line.split('{')[0] == "DCGM_FI_PROF_GR_ENGINE_ACTIVE":
                    gract = line.split(' ')[-1]
                if line.split('{')[0] == "DCGM_FI_DEV_POWER_USAGE":
                    power = line.split(' ')[-1]
    return gract, fbusd, power, profile


