#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Author: Yizheng Huang
Email: yli056@e.ntu.edu.sg
Email: huangyz0918@gmail.com
Date: 9/13/2020
Track all the SLO data performing NLP task from the lightweight distributed system under a simple workload.
The workload is a Poisson process with user specified arrival rate.
The collected metric is combine of client and server settings, timing information (end-to-end latency,
inference latency, etc.), monitor information (CPU utilization, GPU utilization, Memory, etc.):
The result will be saved into MongoDB collection. The collection default name is `test`. You can provide an argument
`-dbn` or `--database_name` to change the collection to save.
Examples:
    usage: slo_tracker_cv_no_scale.py [-h] -b BS -m MODEL [MODEL ...] [--url URL]
                                      [-n NAME] [-dbn DATABASE_NAME] [-r RATE]
                                      [-t TIME] [--text TEXT]
    optional arguments:
      -h, --help            show this help message and exit
      -b BS, --bs BS        frontend batch size
      -m MODEL [MODEL ...], --model MODEL [MODEL ...]
                            A list of names of the used models. For example,
                            resnet18.
      --url URL             The host url of your services. Default to
                            http://localhost:8000.
      -n NAME, --name NAME  The service name you are testing. Default to
                            image_classification.
      -dbn DATABASE_NAME, --database_name DATABASE_NAME
                            The database name you record data to. Default to test.
      -r RATE, --rate RATE  The arrival rate. Default to 5.
      -t TIME, --time TIME  The testing duration. Default to 30.
      --text TEXT           The path to your testing image. Default to
                            ${PROJECT_ROOT}/exp/data/img_bigbang_scene.jpg
Attributes:
    DATA_PATH: Path to the default images for testing.
    SEED: Seed for generating requests arriving follows a Poisson distribution.
Notes:
    You may experience the reported exception: [Errno 24] Too many open files.
    This exception raised probabiliy as a result of too little file open number. Check the number by:
    ````shell script
    ulimit -n
    ```
    Change a larger number:
    ```shell script
    ulimit -n 4096
    ```
"""
import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

from migperf.dcgm_exporter import DCGMMetricCollector
from generator import WorkloadGenerator
from migperf.profiler.utils.misc import consolidate_list_of_dict
from migperf.profiler.utils.request import make_restful_request_from_numpy
# from utils.logger import Printer
from migperf.profiler.utils.pipeline_manager import PreProcessor

DATA_PATH = str(Path(__file__).parent / 'n02124075_Egyptian_cat.jpg')
SEED = 666
start_time = 0
request_num = 0

results = set()

send_time_list = []


# noinspection DuplicatedCode
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bs', help='frontend batch size', type=int, required=True)
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Name of the used models. For example, resnet18.')
    parser.add_argument('--url', type=str, default='http://localhost:50075',
                        help='The host url of your services. Default to http://localhost:50075.')
    parser.add_argument('-T', '--task', type=str, default='image_classification',
                        help='The service name you are testing. Default to image_classification.')
    parser.add_argument('-dbn', '--database_name', type=str, default='test',
                        help='The database name you record data to. Default to test.')
    parser.add_argument('--report-suffix', type=str, default='',
                        help='Suffix to the database name the record data saved.')
    parser.add_argument('-r', '--rate', help='The arrival rate. Default to 5.', type=float, default=5)
    parser.add_argument('-t', '--time', help='The testing duration. Default to 30.', type=float, default=30)
    parser.add_argument('--data', type=str, default=DATA_PATH,
                        help=f'The path to your testing image. Default to {DATA_PATH}')
    parser.add_argument('-P', '--preprocessing', action='store_true', help='Use client preprocessing.')
    # GPU related arguments
    parser.add_argument('-i', '--gpu-id', type=int, default=0, help='GPU ID. Default to 0.')
    parser.add_argument(
        '-gi', '--gpu-instance-id', type=int, default=None,
        help='GPU Instance ID. Specified when MIG is enabled.'
    )
    # experiment settings
    parser.add_argument('--dry-run', action='store_true', help='Dry running the experiment without save result.')
    return parser.parse_args()


def sender(url, request):
    global latency_list

    send_time = time.time()
    response = requests.post(url, **request)
    receive_time = time.time()
    assert response.status_code == 200
    result = response.json()
    response.close()
    latency = receive_time - send_time
    client_server_rtt = latency - result['times']['server_end2end_time']
    result['times'].update({
        'latency': latency,
        'client_server_rtt': client_server_rtt,
    })
    return result


def warm_up(args):
    """Warm up for 100 requests at 10ms each pre GPU worker"""
    url = f'{args.url}/predict'
    with open(args.data, 'rb') as f:
        image = f.read()
    image_np = np.frombuffer(image, dtype=np.uint8)
    if args.preprocessing:
        image_np = PreProcessor.transform_image2torch([image_np]).numpy()[0]
    request = make_restful_request_from_numpy(image_np)
    num = args.bs * 100

    with ThreadPoolExecutor(10) as executor:
        futures = set()
        for i in range(num):
            futures.add(executor.submit(sender, url, request))
            time.sleep(0.01)
        for future in as_completed(futures):
            future.result()


def send_stress_test_data(args):
    """
    send stress testing data.
    """
    global start_time, request_num, send_time_list

    arrival_rate = args.rate
    duration = args.time
    url = f'{args.url}/predict'
    with open(args.data, 'rb') as f:
        image = f.read()
    image_np = np.frombuffer(image, dtype=np.uint8)
    if args.preprocessing:
        image_np = PreProcessor.transform_image2torch([image_np]).numpy()[0]
    request = make_restful_request_from_numpy(image_np)

    send_time_list = WorkloadGenerator.gen_arrival_time(
        duration=duration, arrival_rate=arrival_rate, seed=SEED
    )

    # cut list to a multiple of <BATCH_SIZE>, so that the light-weight system can do full batch prediction
    request_num = len(send_time_list) // args.bs * args.bs
    print(f'Generating {request_num} exadmples')

    start_time = time.time()

    with ThreadPoolExecutor(10) as executor:
        for arrive_time in tqdm(send_time_list[:request_num]):
            results.add(executor.submit(sender, url, request))
            time.sleep(max(arrive_time + start_time - time.time(), 0))


def process_result(args):
    timing_metric_names = [
        'latency', 'client_server_rtt',  # 'batching_time',
        'inference_time', 'postprocessing_time'
    ]
    if not args.preprocessing:
        timing_metric_names.append('preprocessing_time')
    timing_metric_raw_result_dict = defaultdict(list)
    timing_metric_aggr_result_dict = dict()

    raw_result = list()
    fail_count = 0
    for future in as_completed(results):
        try:
            raw_result.append(future.result()['times'])
        except Exception:
            fail_count += 1
            print('.', end='')
            if fail_count % 20 == 0:
                print(fail_count)

    finish_time = time.time()

    for result in raw_result:
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

    # report
    print(f'Failing test number: {fail_count}')

    result = {
        'test_time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'start_time': start_time,
        'arrival_rate': args.rate, 'testing_time': args.time,
        'batch_size': args.bs, 'time_list': send_time_list,
        'model_name': args.model, 'task': args.task,
        'fail_count': fail_count, 'qps': request_num / (finish_time - start_time),
        'client_preprocessing': args.preprocessing,
    }

    result.update(timing_metric_raw_result_dict)
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
    # if MIG is enabled, also obtain sibling GPU instance profile
    if config['mig']['enabled']:
        gpu_instance_profiles = [config['mig']['gpu_instance_profile']]
        for k, v in gpu_metrics_dict.items():
            if k[0] == args.gpu_id and k[1] != args.gpu_instance_id:
                gpu_instance_profiles.append(v['labels'][0]['GPU_I_PROFILE'])
        config['mig']['gpu_instance_profiles'] = gpu_instance_profiles
    result['gpu_model_name'] = config['gpu_static_profile']['modelName']
    result['config'] = config
    return result


if __name__ == '__main__':
    args_ = get_args()
    dcgm_metrics_collector = DCGMMetricCollector()

    print('Testing on:')
    print(f'arrival rate: {args_.rate};', f'testing time: {args_.time};')
    print(f'batch size: {args_.bs};', f'model name: {args_.model}')

    print('Warming up...')
    warm_up(args_)
    print('Testing...')
    dcgm_metrics_collector.start()
    send_stress_test_data(args_)
    print('Finish')

    metrics = process_result(args_)
    dcgm_metrics_collector.stop()
    # save the experiment records to the database and print to the console.
    # TODO: note that you need to change doc_name
    # Printer.add_record_to_database(metrics, db_name='ml_cloud_autoscaler',
    #                                address="mongodb://mongodb.withcap.org:27127/",
    #                                doc_name=args.database_name)
    # temp save to json TODO: manual upload to a DB
    if args_.dry_run:
        print('Dry running, result will not dumped')
        exit(0)

    save_json_file_name = Path(args_.database_name) / (
            '_'.join([
                metrics['gpu_model_name'].replace(' ', '-'),
                metrics["model_name"],
                f'bs{metrics["batch_size"]}',
                f'rate{metrics["arrival_rate"]}',
            ]) + (f'_{args_.report_suffix}' if args_.report_suffix else '') + f'.json'
    )
    save_json_file_name.parent.mkdir(exist_ok=True)
    with open(save_json_file_name, 'w') as f:
        json.dump(metrics, f)
        print(f'result saved successfully as {save_json_file_name}')
