"""
refernce: https://github.com/nvidia/mig-parted
"""
import subprocess


class MIGController:
    """
    In charge of mig partition and recovery
    """
    def __init__(self):
        pass

    def partition(self):
        """
        takes in an assigned partition(gpiID with mig partition),
        if success, returns a data structure contains the deviceIds of all created mig device.
        Return example:
            mig-devices:
                1g.5gb: 'MIG-29f07255-51c0-5691-82c4-cbc57760ff63'
                2g.10gb: 'MIG-ad654d5e-db91-50e1-bcf4-1e9c24f825ad'
                3g.20gb: 'MIG-ea08ed7b-a485-5967-9c78-2fa6e548c43a'
        """
        pass

    def recover(self):
        """
        stop all cuda processes on img devices and then delete all mig devices.
        """
        pass


def reset_mig(gpu_id):
    dci_output = delete_compute_instance(gpu_id)
    dgi_output = delete_gpu_instance(gpu_id)
    return dci_output, dgi_output


def create_mig_profile(gpu_id, mig_profile_id):
    cmd = f'sudo nvidia-smi mig -i {str(gpu_id)} -cgi {str(mig_profile_id)} -C'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.communicate()[0].decode("utf-8")


def delete_compute_instance(gpu_id, gpu_instance_id=None, compute_instance_id=None):
    if gpu_instance_id is None and compute_instance_id is None:
        cmd = f"sudo nvidia-smi mig -i {gpu_id} -dci"
    else:
        cmd = f"sudo nvidia-smi mig -i {gpu_id} -dci -ci {compute_instance_id} -gi {gpu_instance_id}"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.communicate()[0].decode("utf-8")


def delete_gpu_instance(gpu_id, gpu_instance_id=None):
    if gpu_instance_id is None:
        cmd = f"sudo nvidia-smi mig -i {gpu_id} -dgi"
    else:
        cmd = f"sudo nvidia-smi mig -i {gpu_id} -dgi -gi {gpu_instance_id}"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.communicate()[0].decode("utf-8")


def enable_mig_mode(gpu_id):
    cmd = f'nvidia-smi -i {str(gpu_id)} -mig 1'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.communicate()[0].decode("utf-8")


