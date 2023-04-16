"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: Dec 09, 2022

MPS Python wrapper to enable and disable MPS.
"""
import os
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Union


def enable_mps(gpu_ids: Union[int, List[int]] = None):
    """nvidia-cuda-mps-control -d"""
    env = os.environ.copy()
    # Concatenate CUDA_VISIBLE_DEVICES
    if gpu_ids is not None:
        if isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

    # Enable NVIDIA MPS
    subprocess.call(
        ['nvidia-cuda-mps-control', '-d'], 
        stdout=subprocess.DEVNULL,
        env=env,
    )

    return True


def check_mps_status():
    """pidof nvidia-cuda-mps-control"""
    return_code = subprocess.call(
        ['pidof', 'nvidia-cuda-mps-control'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return return_code == 0


def disable_mps(gpu_ids: Union[int, List[int]] = None):
    """echo quit | nvidia-cuda-mps-control"""
    env = os.environ.copy()
    # Concatenate CUDA_VISIBLE_DEVICES
    if gpu_ids is not None:
        if isinstance(gpu_ids, int):
            gpu_ids = [gpu_ids]
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

    p = subprocess.Popen(    
        ['nvidia-cuda-mps-control'], 
        stdin=subprocess.PIPE, 
        stdout=subprocess.DEVNULL,
        env=env,
    )
    p.communicate(input=b'quit')
    
    return True

