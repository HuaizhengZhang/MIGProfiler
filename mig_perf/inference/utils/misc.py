#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 11/3/2020
"""
import re
import subprocess
from typing import Optional


def camelcase_to_snakecase(camel_str):
    """
    Convert string in camel case to snake case.
    References:
        https://www.geeksforgeeks.org/python-program-to-convert-camel-case-string-to-snake-case/
    Args:
        camel_str: String in camel case.
    Returns:
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camel_str)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def consolidate_list_of_dict(l: list, depth=-1):
    if not isinstance(l[0], dict) or depth == 0:
        return l
    d = dict()
    for k in l[0]:
        d[k] = consolidate_list_of_dict([dic[k] for dic in l], depth=depth - 1)
    return d


def get_gpu_device_uuid(gpu_id: int, gpu_mig_device_id: Optional[int] = None):
    """nvidia-smi -L"""
    p = subprocess.Popen(['nvidia-smi', '-L'], stdout=subprocess.PIPE)
    output, err = p.communicate()
    gpu_list_str = output.decode()

    current_gpu_id = None
    for line in gpu_list_str.splitlines():
        # check if the line contains the GPU information.
        # E.g.: GPU 0: NVIDIA A30 (UUID: GPU-bd8c3d28-4b3e-e4ad-650a-4c5a3692b72f)
        match = re.match(r'GPU\s+(\d+)', line)
        if match:
            # Found GPU ID line
            current_gpu_id = int(match.group(1))
            # start from this line, all information is about this GPU ID
            if gpu_mig_device_id is None:
                # Return current GPU UUID
                return line.split('UUID:')[1].strip().rstrip(')')
        elif current_gpu_id == gpu_id:
            # check if the line contains the MIG UUID we cared about
            # E.g.,   MIG 1g.6gb      Device  1: (UUID: MIG-6dd9381e-80bd-5581-9702-563ef12adf3a)
            match = re.match(rf'.*Device\s+{gpu_mig_device_id}', line)
            if match:
                # Found MIG Device ID line
                return line.split('UUID:')[1].strip().rstrip(')')
    # not found
    return None


def get_ids_from_mig_device_id(gpu_id: int, mig_device_id: int):
    """nvidia-smi -i ${GPU_ID} |
      grep -Pzo "\|\s+${GPU_ID}\s+(\d+)\s+(\d+)\s+${MIG_DEVICE_ID}"

    Returns:
        A tuple of GPU instance ID (GI ID) and Compute instance ID (CI ID)
    """
    p = subprocess.Popen(['nvidia-smi', '-i', str(gpu_id)], stdout=subprocess.PIPE)
    output, err = p.communicate()
    gpu_list_str = output.decode()

    for line in gpu_list_str.splitlines():
        match = re.match(rf'\|\s+{gpu_id}\s+(\d+)\s+(\d+)\s+{mig_device_id}', line)
        if match:
            # Found GPU ID line
            gpu_instance_id = int(match.group(1))
            compute_instance_id = int(match.group(2))
            # start from this line, all information is about this GPU ID
            return gpu_instance_id, compute_instance_id
    return None, None
