"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: Dec 09, 2022

MPS Python wrapper to enable and disable MPS.
"""
import subprocess


def enable_mps():
    """nvidia-cuda-mps-control -d"""
    # Is MPS already enabled
    if check_mps_status():
        return

    # Enable NVIDIA MPS
    subprocess.call(
        ['nvidia-cuda-mps-control', '-d'], 
        stdout=subprocess.DEVNULL,
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


def disable_mps():
    """echo quit | nvidia-cuda-mps-control"""
    if not check_mps_status():
        return

    p = subprocess.Popen(    
        ['nvidia-cuda-mps-control'], 
        stdin=subprocess.PIPE, 
        stdout=subprocess.DEVNULL,
    )
    p.communicate(input=b'quit')
    
    return True

