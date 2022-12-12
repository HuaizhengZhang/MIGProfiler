#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: Dec 5, 2022
"""
import time
from collections import defaultdict
from threading import Thread

import requests
from prometheus_client.parser import text_string_to_metric_families


def dcgm_gpu_metric_parser(metrics: str):
    # TODO: change GPU Instance ID -> MIG Device ID. GPU Instance ID is not determined by device order
    # gpu_id -> { gpu_instance_id -> metrics }
    # if no gpu instance (MIG not enabled), gpu_instance_id is None
    gpu_metrics_dict = defaultdict(dict)
    for family in text_string_to_metric_families(metrics):
        for sample in family.samples:
            name, labels, value = sample.name, sample.labels, sample.value
            # build id
            gpu_id = int(labels['gpu'])
            if labels.get('GPU_I_ID', None):
                gpu_instance_id = int(labels['GPU_I_ID'])
            else:
                gpu_instance_id = None

            if not gpu_metrics_dict[gpu_id, gpu_instance_id]:
                gpu_metrics_dict[gpu_id, gpu_instance_id]['labels'] = labels
            gpu_metrics_dict[gpu_id, gpu_instance_id][name] = value

    return gpu_metrics_dict


class DCGMMetricCollector(object):
    def __init__(self, dcgm_url='http://0.0.0.0:9400/metrics'):
        self.dcgm_url = dcgm_url

        self._thread = Thread(target=self.runner)
        self.is_running = False
        self.gpu_metrics_list = list()

    def runner(self):
        while self.is_running:
            metrics = requests.get(self.dcgm_url).text
            data_collected_time = time.time()
            metrics = dcgm_gpu_metric_parser(metrics)
            metrics['time'] = data_collected_time
            self.gpu_metrics_list.append(metrics)
            time.sleep(1)

    def start(self):
        self.is_running = True
        self._thread.start()

    def stop(self):
        self.is_running = False
        self._thread.join()


if __name__ == '__main__':
    collector = DCGMMetricCollector()
    collector.start()
    time.sleep(3)
    collector.stop()
    print(collector.gpu_metrics_list)
