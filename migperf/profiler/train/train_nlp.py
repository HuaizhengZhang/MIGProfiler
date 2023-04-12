#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: Dec 14, 2022
"""
import argparse
import json
import os
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from migperf.dcgm_exporter import DCGMMetricCollector
from migperf.profiler.utils.data_hub import load_amazon_review_data
from migperf.profiler.utils.misc import get_ids_from_mig_device_id, get_gpu_device_uuid, consolidate_list_of_dict
from migperf.profiler.utils.model_hub import load_pytorch_model

start_time = 0
raw_results = list()


def get_args():
    parser = argparse.ArgumentParser(description='NLP model training')
    parser.add_argument('-T', '--task', type=str, default='single_label_classification',
                        help='The service name you are testing. Default to image_classification.')
    parser.add_argument('-b', '--bs', help='training batch size', type=int, default=256)
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Name of the used models. For example, bert-base-cased.')
    parser.add_argument('--seq_len', type=int, default=64, help='Max sequence length')
    parser.add_argument('-n', '--max_train_steps', type=int, required=True, help='Total number of batches to test.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--num_classes', default=5, type=int, help='num of class in the model')
    # GPU related arguments
    parser.add_argument(
        '-i', '--gpu-id', type=int, default=0,
        help='GPU ID. Default to 0. This is only for record purpose.'
    )
    parser.add_argument(
        '-mi', '--mig-device-id', type=int, default=None,
        help='GPU Instance ID. Specified when MIG is enabled. This is only for record purpose.'
    )
    # experiment settings
    parser.add_argument('-dbn', '--database_name', type=str, default='test',
                        help='The database name you record data to. Default to test.')
    parser.add_argument('--report-suffix', type=str, default='',
                        help='The suffix of the record saving file name')
    parser.add_argument('--dry-run', action='store_true', help='Dry running the experiment without save result.')
    args = parser.parse_args()
    args.device_uuid = get_gpu_device_uuid(args.gpu_id, args.mig_device_id)
    assert args.device_uuid is not None, \
        f'Cannot find device UUID of GPU ID: {args.gpu_id}, MIG Device ID: {args.mig_device_id}'
    args.gpu_instance_id, args.compute_instance_id = get_ids_from_mig_device_id(args.gpu_id, args.mig_device_id)
    return args


def warm_up(args):
    """Warm up for 100 batches each pre GPU worker"""
    num = 100
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            if step == num:
                break
            for k, v in batch.items():
                batch[k] = v.cuda()
            model(**batch)
    torch.cuda.synchronize()


def train_func(args):
    global start_time

    model.train()
    step_start_time = start_time = time.time()
    step_num = args.max_train_steps or len(train_dataloader)
    for step, batch in enumerate(tqdm(train_dataloader, total=step_num)):
        if args.max_train_steps and step >= args.max_train_steps:
            break

        for k, v in batch.items():
            batch[k] = v.cuda()
        data_process_time = time.time()
        labels = batch['labels']
        output = model(**batch)
        loss = criterion(output.logits, labels)
        torch.cuda.synchronize()
        forward_time = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_end_time = backward_time = time.time()
        raw_results.append({
            'step_latency': step_end_time - step_start_time,
            'data_process_time': data_process_time - step_start_time,
            'forward_time': forward_time - data_process_time,
            'backward_time': backward_time - forward_time,
        })
        step_start_time = step_end_time


def process_result(args):
    timing_metric_names = [
        'step_latency', 'data_process_time',
        'forward_time', 'backward_time',
    ]
    timing_metric_raw_result_dict = defaultdict(list)
    timing_metric_aggr_result_dict = dict()

    finish_time = time.time()

    for result in raw_results:
        for metric_name in timing_metric_names:
            timing_metric_raw_result_dict[metric_name].append(result[metric_name])

    for metric_name, raw_result in timing_metric_raw_result_dict.items():
        mean = np.mean(raw_result)
        std = np.std(raw_result)
        p50 = np.percentile(raw_result, 50)
        p95 = np.percentile(raw_result, 95)
        p99 = np.percentile(raw_result, 99)

        timing_metric_aggr_result_dict[metric_name + '_mean'] = mean
        timing_metric_aggr_result_dict[metric_name + '_std'] = std
        timing_metric_aggr_result_dict[metric_name + '_p50'] = p50
        timing_metric_aggr_result_dict[metric_name + '_p95'] = p95
        timing_metric_aggr_result_dict[metric_name + '_p99'] = p99

    result = {
        'test_time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'start_time': start_time,
        'train_steps': args.max_train_steps, 'batch_size': args.bs, 'sequence_length': args.seq_len, 
        'model_name': args.model, 'task': args.task, 
        'learning_rate': args.lr, 'weight_decay': args.weight_decay,
        'qps': args.max_train_steps * args.bs / (finish_time - start_time),
    }

    result.update(timing_metric_aggr_result_dict)
    result.update(timing_metric_raw_result_dict)
    gpu_metrics_list = deepcopy(dcgm_metrics_collector.gpu_metrics_list)
    gpu_metrics_dict = consolidate_list_of_dict(gpu_metrics_list, depth=2)
    # gpu_label_example = {
    #     'gpu': '0', 'UUID': 'GPU-bd8c3d28-4b3e-e4ad-650a-4c5a3692b72f', 'device': 'nvidia0',
    #     'modelName': 'NVIDIA A30', 'Hostname': '2e140b568f0c',
    #     'GPU_I_PROFILE': '4g.24gb', 'GPU_I_ID': '0',
    # }
    gpu_labels: dict = gpu_metrics_dict[args.gpu_id, args.gpu_instance_id].pop('labels')[0]
    result['metrics'] = gpu_metrics_dict[args.gpu_id, args.gpu_instance_id]
    # patch GPU metrics back
    gpu_metrics_dict[args.gpu_id, args.gpu_instance_id]['labels'] = [gpu_labels]

    # export config
    config = {
        'client_args': vars(args),
        'gpu_static_profile': gpu_labels,
        'mig': {
            'enabled': gpu_labels.get('GPU_I_ID', None) is not None,
            'gpu_instance_id': gpu_labels.get('GPU_I_ID', None),
            'gpu_instance_profile': gpu_labels.get('GPU_I_PROFILE', None),
        },
    }
    # if MIG is enabled, also obtain sibling GPU instance profile
    if config['mig']['enabled']:
        gpu_instance_profiles = list()
        for k, v in gpu_metrics_dict.items():
            if k[0] == args.gpu_id:
                gpu_instance_profiles.append(v['labels'][0]['GPU_I_PROFILE'])
        config['mig']['gpu_instance_profiles'] = gpu_instance_profiles
    result['gpu_model_name'] = config['gpu_static_profile']['modelName']
    result['config'] = config
    return result


if __name__ == '__main__':
    args_ = get_args()
    # Mask out other cuda devices
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args_.device_uuid
    cudnn.benchmark = True
    dcgm_metrics_collector = DCGMMetricCollector()

    print('Testing on:')
    print(f'num of test training steps: {args_.max_train_steps};')
    print(f'batch size: {args_.bs};', f'model name: {args_.model}')

    print('Prepare dataset...')
    tokenizer = AutoTokenizer.from_pretrained(args_.model)
    train_dataloader, val_dataloader = load_amazon_review_data(
        batch_size=args_.bs, max_seq_len=args_.seq_len, tokenizer=tokenizer,
    )

    print(f'Load {args_.model} model...')
    model = load_pytorch_model(model_name=args_.model, num_labels=args_.num_classes).cuda()

    print('Setup optimizer')
    optimizer = torch.optim.AdamW(model.parameters(), args_.lr, weight_decay=args_.weight_decay)
    print('Setup loss')
    criterion = nn.CrossEntropyLoss().cuda()

    print('Warming up...')
    warm_up(args_)
    print('Training...')
    dcgm_metrics_collector.start()
    train_func(args_)
    print('Finish')
    metrics = process_result(args_)
    dcgm_metrics_collector.stop()
    # save the experiment records to the database and print to the console.
    if args_.dry_run:
        print('Dry running, result will not dumped')
        exit(0)

    save_json_file_name = Path(
        args_.database_name) / (
                                  '_'.join([
                                      metrics['gpu_model_name'].replace(' ', '-'),
                                      metrics["model_name"],
                                      f'bs{metrics["batch_size"]}',
                                      f'seq{metrics["sequence_length"]}',
                                      f'lr{metrics["learning_rate"]}',
                                  ]) + (f'_{args_.report_suffix}' if args_.report_suffix else '') + '.json'
                              # f'_{metrics["test_time"]}.json'
                          )
    save_json_file_name.parent.mkdir(exist_ok=True, parents=True)
    with open(save_json_file_name, 'w') as f:
        json.dump(metrics, f)
        print(f'result saved successfully as {save_json_file_name}')

