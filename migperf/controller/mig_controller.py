"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: Dec 09, 2022

MIG controller Python wrapper to create/destroy instances, query status for MIG-enabled GPU devices.

References: https://github.com/nvidia/mig-parted
"""
import re
import subprocess
from typing import List, Union


class MIGController(object):
    CREATE_GI_PATTERN = re.compile(r'.+GPU instance ID\s+(\d+).+\s+(\d+).+(MIG\s+\d+g\.\d+gb)\s+\(ID\s+(\d+)')
    CREATE_CI_PATTERN = re.compile(
        r'.+compute instance ID\s+(\d+).+\s(\d+).+\s(\d+).+(MIG\s+\d+g\.\d+gb|MIG\s+\d+c\.\d+\.\d+gb)\s+\(ID\s+(\d+)'
    )
    GI_STATUS_PATTERN = re.compile(r'\|\s+(\d+)\s+(MIG\s+\d+g\.\d+gb)\s+(\d+)\s+(\d+)\s+(\d+)\:(\d+)')
    CI_STATUS_PATTERN = re.compile(
        r'\|\s+(\d+)\s+(\d+)\s+(MIG\s+\d+g\.\d+gb|MIG\s+\d+c\.\d+g\.\d+gb)\s+(\d+)\s+(\d+)\s+(\d+)\:(\d+)'
    )

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
            current_state, pending_state = (
                current_state == 'Enabled'), (pending_state == 'Enabled')
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
    def create_gpu_instance(cls, gi_profiles: Union[str, List[str]], gpu_id: int = None, create_ci: bool = False):
        """Create GPU instance on MIG-enabled GPU device.
        The function is equivalant to executing the command: 
        :code:`sudo nvidia-smi mig -i ${gpu_id} -cgi ${gi_profiles}`
        
        Args:
            gi_profiles (str or list of str): Profile tuple or a list of profile 
                tuple. A profile tuple consists of 1. a profile name or ID and 
                2. an optional placement specifier, which consists of a colon and a 
                placement start index.
                For example, 
                    :code:`1g.10gb`, or :code:`15`: GI with profile 1/7 SM + 10gb memory.
                    :code:`1g.10gb:0`, or :code:`15:0`: GI with profile 1/7 SM + 10gb memory placed at 0.
            gpu_id (int, optional): ID of the specified GPU to create the GPU instance.
                Not specifying :code:`gpu_id` will result in create GPU instances on
                every available GPUs.
            create_ci (bool, optional): Create the default* compute instance 
                after each GPU instance creation. Default to `False`.
        Returns:
            list of dict: A list of created GPU instance status.
                The dictionary contains: :code:`gpu_id`, :code:`name`, 
                :code:`profile_id`, and :code:`gi_id` (GPU instance ID).
        Raises:
            ValueError: If creating of the GPU instance fails.
        """
        if isinstance(gi_profiles, list):
            gi_profiles = ','.join(gi_profiles)

        cmd = ['sudo', 'nvidia-smi', 'mig', '-i', str(gpu_id), '-cgi', gi_profiles]
        if create_ci:
            # Also create the corresponding compute Instances (CI)
            cmd.append('-C')
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, encoding='utf-8')
        output, _ = p.communicate()
        # Check if there are something failed
        if 'Failed' in output or 'No' in output:
            raise ValueError(
                f'Failed to create GPU instance when executing the command: {" ".join(cmd)}\n'
                f'{output}'
            )

        gi_status_list = list()
        for line in output.splitlines():
            match_groups = cls.CREATE_GI_PATTERN.match(line)
            if match_groups is not None:
                gi_id, g_id, name, profile_id = match_groups.groups()
                gi_status_list.append({
                    'gpu_id': int(g_id), 'name': name, 'profile_id': int(profile_id), 'gi_id': int(gi_id),
                })

        return gi_status_list

    @classmethod
    def create_compute_instance(cls, ci_profiles: Union[str, List[str]] = None, gpu_id: int = None, gi_id: int = None):
        """Create compute instance on MIG-enabled GPU device.
        The function is equivalant to executing the command: 
        :code:`sudo nvidia-smi mig -i ${gpu_id} -gi ${gi_id} -cci ${ci_profiles}`
        
        Args:
            ci_profiles (str or list of str): Profile name / ID or a list of profile
                names / IDs. If no profile name or ID is given, then the default*
                compute instance profile ID will be used.
                For example, 
                    :code:`1c.1g.10gb`, or :code:`0`: use 1/7 SM.
            gpu_id (int, optional): ID of the specified GPU to create the compute instance.
                Not specifying :code:`gpu_id` will result in create compute instances on
                every available GPUs.
            gi_id (bool, optional): ID of the specified GPU instance to create 
                the compute instance. Not specifying :code:`gi_id` will result in
                create compute instances on every avaliable GPU instances.
        Returns:
            list of dict: A list of created GPU instance status.
                The dictionary contains: :code:`gpu_id`, :code:`name`, 
                :code:`profile_id`, and :code:`gi_id` (GPU instance ID).
        Raises:
            ValueError: If creating of the GPU instance fails.
        """
        if isinstance(ci_profiles, list):
            ci_profiles = ','.join(ci_profiles)

        cmd = ['sudo', 'nvidia-smi', 'mig', '-cci']
        if gpu_id is not None:
            cmd.extend(['-i', str(gpu_id)])
        if gi_id is not None:
            cmd.extend(['-gi', str(gi_id)])
        if ci_profiles is not None:
            cmd.append(ci_profiles)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, encoding='utf-8')
        output, _ = p.communicate()
        # Check if there are something failed
        if 'Failed' in output or 'No' in output:
            raise ValueError(
                f'Failed to create compute instance when executing the command: {" ".join(cmd)}\n'
                f'{output}'
            )

        ci_status_list = list()
        for line in output.splitlines():
            match_groups = cls.CREATE_CI_PATTERN.match(line)
            if match_groups is not None:
                ci_id, g_id, gi_id, name, profile_id = match_groups.groups()
                ci_status_list.append({
                    'gpu_id': int(g_id), 'gi_id': int(gi_id), 'name': name, 'ci_id': int(ci_id), 
                    'profile_id': int(profile_id),
                })
    
        return ci_status_list

    @classmethod
    def check_gpu_instance_status(cls, gpu_id: int = None):
        """sudo nvidia-smi mig -lgi -i ${gpu_id}
        
        Returns: list of dict.
            A list of GPU Instance status. Example: [{'gpu': 0, 'gi_id': 13, 'name': 'MIG 1g.10gb', 
                'profile_id': 19, 'ci_id': 1, 'placement': {'start': 0, 'size': 1}}]
        """
        cmd = ['sudo', 'nvidia-smi', 'mig', '-lgi']
        if gpu_id is not None:
            cmd.extend(['-i', str(gpu_id)])

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            encoding='utf-8',
        )
        output, _ = p.communicate()
        # parse output CSV string
        gi_status_list = list()
        for line in output.splitlines():
            match_groups = cls.GI_STATUS_PATTERN.match(line)
            if match_groups:
                g_id, name, profile_id, gi_id, placement_start, placement_size = match_groups.groups()
                gi_status_list.append({
                    'gpu_id': int(g_id), 'name': name, 'profile_id': int(profile_id), 'gi_id': int(gi_id),
                    'placement': {'start': int(placement_start), 'size': int(placement_size)}
                })
        
        return gi_status_list
    
    @classmethod
    def check_compute_instance_status(cls, gpu_id: int = None, gi_id: int = None):
        """sudo nvidia-smi -lci -i ${gpu_id} -gi ${gi_id}
        
        Returns: list of dict.
            A list of Compute Instance status. Example: [{'gpu': 0, 'name': 'MIG 1g.10gb', 'profile_id': 19,
                  'gi_id': 11, 'placement': {'start': 4, 'size': 1}}]
        """
        cmd = ['sudo', 'nvidia-smi', 'mig', '-lci']
        if gpu_id is not None:
            cmd.extend(['-i', str(gpu_id)])
        if gi_id is not None:
            cmd.extend(['-gi', str(gi_id)])

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            encoding='utf-8',
        )
        output, _ = p.communicate()
        # parse output CSV string
        ci_status_list = list()
        for line in output.splitlines():
            match_groups = cls.CI_STATUS_PATTERN.match(line)
            if match_groups:
                g_id, gi_id, name, profile_id, ci_id, placement_start, placement_size = match_groups.groups()
                ci_status_list.append({
                    'gpu_id': int(g_id), 'gi_id': int(gi_id), 'name': name, 'profile_id': int(profile_id), 
                    'ci_id': int(ci_id), 'placement': {'start': int(placement_start), 'size': int(placement_size)}
                })
        return ci_status_list

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
    
    @classmethod
    def list_gpu_instance_profiles(gpu_id: int = None):
        """sudo nvidia-smi -lgip -i ${gpu_id}"""
        # TODO: list all GPU instance profiles
        raise NotImplementedError()
    
    @classmethod
    def list_gpu_instance_possible_placements(gpu_id: int = None):
        """sudo nvidia-smi -lgipp -i ${gpu_id}"""
        # TODO: list all GPU instance possible placement
        raise NotImplementedError()
    
    @classmethod
    def list_compute_instance_profiles(gpu_id: int = None, gi_id: int = None):
        """sudo nvidia-smi -lcip -i ${gpu_id} -gi ${gi_id}"""
        # TODO: list all compute instance profiles
        raise NotImplementedError()


if __name__ == '__main__':
    mig_controller = MIGController()
    mig_controller.enable_mig(0)
    print(mig_controller.check_mig_status(0))
    gi_instances = mig_controller.create_gpu_instance(gi_profiles='1g.10gb,1g.10gb', gpu_id=0)
    print(gi_instances)
    print(mig_controller.check_gpu_instance_status(gpu_id=0))
    ci_instances = mig_controller.create_compute_instance(ci_profiles='1g.10gb')
    print(ci_instances)
    print(mig_controller.check_compute_instance_status(gpu_id=0))
    mig_controller.destroy_compute_instance(gpu_id=0)
    mig_controller.destroy_gpu_instance(gpu_id=0)
    mig_controller.disable_mig(0)
