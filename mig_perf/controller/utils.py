import subprocess


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



