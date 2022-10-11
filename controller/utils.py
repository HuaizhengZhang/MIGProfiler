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


def get_mig_devices(gpu_id):
    cmd = f"nvidia-smi -L"
    mig_devices = []
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, unused_err = p.communicate(timeout=10)
    output = output.decode("utf-8")
    find_gpu = False
    for line in output.splitlines():
        if line.strip().split(':')[0].split(' ')[0] == 'GPU':
            if line.strip().split(':')[0].split(' ')[1] == str(gpu_id):
                find_gpu = True
            else:
                find_gpu = False
                continue
        if find_gpu and line.strip().split(' ')[0] == 'MIG':
            mig_devices.append(
                {
                    'mig_name': line.strip().split(' ')[1],
                    'uuid': line.strip().split(':')[-1].strip().split(')')[0],
                    'instance_id': line.strip().split(':')[0].split(' ')[-1]
                }
            )
    return mig_devices


