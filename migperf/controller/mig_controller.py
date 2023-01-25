"""
refernce: https://github.com/nvidia/mig-parted
"""
import subprocess
import os
from pathlib import Path
from typing import List, Union

SCRIPT_MIGRECONF = str(Path(os.getcwd())/"mig_reconfigure.sh")


class MIGController:
    @classmethod
    def enable_mig(cls, gpu_id: int = None):
        """sudo nvidia-smi -i ${gpu_id} -mig 1"""
        cmd = ['sudo', 'nvidia-smi', '-mig', '1']
        if gpu_id is not None:
            # Enable NVIDIA MIG with specific GPU
            cmd.extend(['-i', str(gpu_id)])
        # Enable NVIDIA MIG
        return subprocess.call(cmd)

    @classmethod
    def check_mig_status(cls, gpu_id: int = None):
        """Execute command: nvidia-smi --query-gpu=mig.mode.current,mig.mode.pending --format=csv,noheader
        
        Returns:
            A list of tuple, contains all GPU's current MIG status and pending MIG status if `gpu_id` is not specified.
            Or a tuple contains the specified GPU's current MIG status and pending MIG status if `gpu_id` is provided.
        """
        p = subprocess.Popen(
            [
                'nvidia-smi', '--query-gpu=mig.mode.current,mig.mode.pending', 
                '--format=csv,noheader',
            ],
            stdout=subprocess.PIPE,
            encoding='utf-8',
        )
        output, _ = p.communicate()
        # parse output CSV string
        mig_status_list = list()
        for line in output.splitlines():
            current_state, pending_state = line.split(', ')
            current_state, pending_state = (current_state == 'Enabled'), (pending_state == 'Enabled')
            mig_status_list.append((current_state, pending_state))
        # select the intereted GPU ID
        if gpu_id is not None:
            return mig_status_list[gpu_id]
        else:
            return mig_status_list

    @classmethod
    def disable_mig(cls, gpu_id: int = None):
        """Execute command: sudo nvidia-smi -i ${gpu_id} -mig 0"""
        cmd = ['sudo', 'nvidia-smi', '-mig', '0']
        if gpu_id is not None:
            # Disable NVIDIA MIG with specified GPU
            cmd.extend(['-i', str(gpu_id)])
        # Disable NVIDIA MIG
        return subprocess.call(cmd)
    
    @classmethod
    def create_gpu_instance(cls, gpu_id: int, i_profiles: Union[str, List[str]], create_ci: bool = False):
        """sudo nvidia-smi mig -i ${gpu_id} -cgi ${i_profiles}"""
        if isinstance(i_profiles, list):
            i_profiles = ','.join(i_profiles)
        
        cmd = ['sudo', 'nvidia-smi', 'mig', '-i', str(gpu_id), '-cgi', i_profiles]
        if create_ci:
            # Also create the corresponding Compute Instances (CI)
            cmd.append('-C')
        return subprocess.call(cmd)
    
    @classmethod
    def create_compute_instance(cls, gi_id: int, ci_profiles: Union[str, List[str]]):
        """sudo nvidia-smi mig -gi ${gi_id} -cci ${ci_profiles}"""
        if isinstance(ci_profiles, list):
            ci_profiles = ','.join(ci_profiles)
        
        cmd = ['sudo', 'nvidia-smi', 'mig', '-gi', str(gi_id), '-cci', ci_profiles]
        return subprocess.call(cmd)

    @classmethod
    def destroy_gpu_instance(cls, gpu_id: int = None, gi_ids: Union[int, List[int]] = None):
        """sudo nvidia-smi mig -dgi -gi ${gi_id} -i ${gpu_id}"""
        cmd = ['sudo', 'nvidia-smi', 'mig', '-dgi']
        if isinstance(gi_ids, list):
            gi_ids = ','.join(gi_ids)
        if gi_ids is not None:
            cmd.extend(['-gi', str(gi_ids)])
        if gpu_id is not None:
            cmd.extend(['-i', str(gpu_id)])
        return subprocess.call(cmd)
    
    @classmethod
    def destroy_compute_instance(cls, gpu_id: int = None, gi_id: int = None, ci_ids: Union[int, List[int]] = None):
        """sudo nvidia-smi mig -dci -gi ${gi_id} -ci ${ci_ids} -i ${gpu_id}"""
        cmd = ['sudo', 'nvidia-smi', 'mig', '-dci']

        if isinstance(ci_ids, list):
            ci_ids = ','.join(ci_ids)
        if ci_ids is not None:
            cmd.extend(['-ci', str(ci_ids)])
        if gi_id is not None:
            cmd.extend(['-gi', str(gi_id)])
        if gpu_id is not None:
            cmd.extend(['-i', str(gpu_id)])
        return subprocess.call(cmd)


if __name__ == '__main__':
    mig_controller = MIGController()
    mig_controller.enable_mig(0)
    print(mig_controller.check_mig_status(0))
    mig_controller.create_gpu_instance(gpu_id=0, i_profiles='1g.10gb,1g.10gb', create_ci=True)
    mig_controller.destroy_compute_instance(gpu_id=0)
    mig_controller.destroy_gpu_instance(gpu_id=0)
    mig_controller.disable_mig(0)

