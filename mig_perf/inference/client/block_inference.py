#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: Dec 8, 2022
"""
import argparse
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch.cuda
from tqdm import trange

from client.monitor import DCGMMetricCollector
from utils.misc import consolidate_list_of_dict
from utils.model_hub import load_pytorch_model
from utils.pipeline_manager import PreProcessor

DATA_PATH = str(Path(__file__).parent / 'n02124075_Egyptian_cat.jpg')

latency_list = list()
start_time = 0
finish_time = 0


def get_args():
    parser = argparse.ArgumentParser(description='Blocked model inference')
    parser.add_argument('-b', '--bs', help='frontend batch size', type=int, required=True)
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Name of the used models. For example, resnet18.')
    parser.add_argument('-T', '--task', type=str, default='image_classification',
                        help='The service name you are testing. Default to image_classification.')
    parser.add_argument('-n', '--num_batches', type=int, required=True, help='Total number of batches to test.')
    parser.add_argument('--data', type=str, default=DATA_PATH,
                        help=f'The path to your testing image. Default to {DATA_PATH}')
    # GPU related arguments
    parser.add_argument('--device-id', type=str, default='0', help='Device ID to pass for CUDA_VISIBLE_DEVICES.')
    parser.add_argument(
        '-i', '--gpu-id', type=int, default=0,
        help='GPU ID. Default to 0. This is only for record purpose.'
    )
    parser.add_argument(
        '-gi', '--gpu-instance-id', type=int, default=None,
        help='GPU Instance ID. Specified when MIG is enabled. This is only for record purpose.'
    )
    # experiment settings
    parser.add_argument('-dbn', '--database_name', type=str, default='test',
                        help='The database name you record data to. Default to test.')
    parser.add_argument('--dry-run', action='store_true', help='Dry running the experiment without save result.')
    return parser.parse_args()


def warm_up(args):
    """Warm up for 100 batches each pre GPU worker"""
    with open(args.data, 'rb') as f:
        image = f.read()
    image_np = np.frombuffer(image, dtype=np.uint8)
    image_tensor = PreProcessor.transform_image2torch([image_np] * args.bs).cuda()
    num = 100
    for _ in range(num):
        model(image_tensor)
    torch.cuda.synchronize()


def test_block_inference(args):
    """Run inference test"""
    global start_time, finish_time
    with open(args.data, 'rb') as f:
        image = f.read()
    image_np = np.frombuffer(image, dtype=np.uint8)
    image_tensor = PreProcessor.transform_image2torch([image_np] * args.bs).cuda()

    start_time = time.time()
    for _ in trange(args.num_batches):
        start = time.time()
        model(image_tensor)
        torch.cuda.synchronize()
        latency_list.append(time.time() - start)
    finish_time = time.time()


def process_result(args):
    timing_metric_aggr_result_dict = dict()

    mean = np.mean(latency_list)
    std = np.std(latency_list)
    p50 = np.percentile(latency_list, 50)
    p95 = np.percentile(latency_list, 95)
    p99 = np.percentile(latency_list, 99)

    timing_metric_aggr_result_dict['latency_mean'] = mean
    timing_metric_aggr_result_dict['latency_std'] = std
    timing_metric_aggr_result_dict['latency_p50'] = p50
    timing_metric_aggr_result_dict['latency_p95'] = p95
    timing_metric_aggr_result_dict['latency_p99'] = p99

    # report
    result = {
        'test_time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'start_time': start_time,
        'num_test_batches': args.num_batches, 'batch_size': args.bs, 'model_name': args.model, 'task': args.task,
        'qps': args.num_batches * args.bs / (finish_time - start_time),
        'latency': latency_list,
    }

    result.update(timing_metric_aggr_result_dict)
    gpu_metrics_list = deepcopy(dcgm_metrics_collector.gpu_metrics_list)
    gpu_metrics_dict = consolidate_list_of_dict(gpu_metrics_list, depth=2)
    # gpu_label_example = {
    #     'gpu': '0', 'UUID': 'GPU-bd8c3d28-4b3e-e4ad-650a-4c5a3692b72f', 'device': 'nvidia0',
    #     'modelName': 'NVIDIA A30', 'Hostname': '2e140b568f0c',
    #     'GPU_I_PROFILE': '4g.24gb', 'GPU_I_ID': '0',
    # }
    gpu_labels: dict = gpu_metrics_dict[args.gpu_id, args.gpu_instance_id].pop('labels')[0]
    result['metrics'] = gpu_metrics_dict[args.gpu_id, args.gpu_instance_id]

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
    result['gpu_model_name'] = config['gpu_static_profile']['modelName']
    result['config'] = config
    return result


if __name__ == '__main__':
    args_ = get_args()
    # Mask out other cuda devices
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args_.device_id
    dcgm_metrics_collector = DCGMMetricCollector()

    print('Testing on:')
    print(f'num of test batches: {args_.num_batches};')
    print(f'batch size: {args_.bs};', f'model name: {args_.model}')

    print(f'Load {args_.model} model...')
    model = load_pytorch_model(model_name=args_.model).cuda()
    print('Warming up...')
    warm_up(args_)
    print('Testing...')
    dcgm_metrics_collector.start()
    test_block_inference(args_)
    print('Finish')
    metrics = process_result(args_)
    dcgm_metrics_collector.stop()
    # save the experiment records to the database and print to the console.
    if args_.dry_run:
        print('Dry running, result will not dumped')
        exit(0)

    save_json_file_name = Path(
        args_.database_name) / (
                                  '_'.join([metrics["model_name"], f'bs{metrics["batch_size"]}']) +
                                  f'_{metrics["test_time"]}.json'
                          )
    save_json_file_name.parent.mkdir(exist_ok=True)
    with open(save_json_file_name, 'w') as f:
        json.dump(metrics, f)
        print(f'result saved successfully as {save_json_file_name}')
