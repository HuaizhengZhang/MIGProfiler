#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: Apr 16, 2023

Run MIG profiling job with configuration YAML file.
"""
from .mig_controller import MIGController
from .mps_controller import enable_mps


def config_gpu_device(gpu_configs: list):
    """
    Configure GPU device with configuration YAML file.

    Args:
        gpu_configs (list): A list of GPU device configuration.

    Examples:
        gpu_config example YAML file: example_config.yaml
        ```yaml
        gpus:
          - id: 0
            mig: true
            mps: false
            devices:
              - gi_profile: 1g.10gb
                task: cv_train
        ```
        >>> import yaml
        >>> gpu_configs = yaml.safe_load('example_config.yaml')['gpus']
        >>> config_gpu_device(gpu_configs)
    """
    # Enable MPS for all required GPUs
    mps_gpu_ids = [gpu_config['id'] for gpu_config in gpu_configs if gpu_config['mps']]
    enable_mps(mps_gpu_ids)

    # Configure MIG for all required GPUs
    # 1. Enable MIG
    # 2. Configure MIG device
    mig_controller = MIGController()
    for gpu_config in gpu_configs:
        gpu_id = gpu_config['id']
        mig_devices = gpu_config['devices']
        if gpu_config['mig']:
            mig_controller.enable_mig(gpu_id)
            for mig_device in mig_devices:
                mig_controller.create_gpu_instance(mig_device['gi_profile'], gpu_id=gpu_id, create_ci=True)


def run_job(config: dict):
    """Run MIG profiling jobs on specific MIG devices with configuration YAML file.

    Args:
        config (dict): A dict of MIG profiling job configuration.

    Examples:
        job_config example YAML file: example_config.yaml
        ```yaml
        gpus:
        - id: 0
          mig: true
          mps: false
          devices:
            - gi_profile: 1g.10gb
              job: cv_train
        job_configs:
            - name: cv_train
              type: train
              ml_task: image_classification
              dataset:
                name: places365_standard
                num_classes: 365
              model:
                name: resnet50
                pretrained: true
                loss: cross_entropy
              optimizer:
                name: sgd
                lr: 0.01
                momentum: 0.9
                weight_decay: 0.0001
              batch_size: 64
              max_train_steps: 100
        ```
    """
    # Parse job execution command from config
    job_configs = config['job_configs']
    jobs = dict()
    for job_config in job_configs:
        task_type = job_configs['type']
        ml_task = job_configs['ml_task']

        if task_type == 'train' and ml_task == 'image_classification':
            # python train/train_cv.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${max_train_steps}" \
            #     -i "${GPU_ID}" -mi 0 -dbn "${EXP_SAVE_DIR}/${MIG_PROFILE}"
            entry_point = 'migperf/profiler/train/train_cv.py'
            args = [
                f'-b={job_config["batch_size"]}',
                f'-m={job_config["model"]["name"]}',
                f'-n={job_config["max_train_steps"]}',
            ]
            jobs[job_config['name']] = {
                'entry_point': entry_point,
                'args': args,
            }

    # Run jobs on MIG devices
