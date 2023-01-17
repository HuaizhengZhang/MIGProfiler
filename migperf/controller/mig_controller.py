"""
refernce: https://github.com/nvidia/mig-parted
"""
import re
import subprocess
import os
from pathlib import Path
from typing import Optional

SCRIPT_MIGRECONF = str(Path(os.getcwd())/"mig_reconfigure.sh")


class MIGPerfController:
    def __init__(self):
        pass

    def reconfigure_mig(gpu_id: int, new_profile: str):
        cmd = f"bash {SCRIPT_MIGRECONF} {str(gpu_id)} {new_profile}"
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return p.communicate()[0].decode("utf-8")

    def enable_mig_mode(gpu_id):
        cmd = f'nvidia-smi -i {str(gpu_id)} -mig 1'
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return p.communicate()[0].decode("utf-8")

