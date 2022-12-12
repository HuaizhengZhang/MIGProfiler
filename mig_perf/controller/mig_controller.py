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
