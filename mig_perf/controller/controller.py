"""
refernce: https://github.com/nvidia/mig-parted
"""
import logging
import subprocess


def profile_plan(gpu_id, mig_profiles):

    def decorator(f):
        def inner(*args, **kwargs):
            # enable gpu mig mode
            enable_mig_out = enable_mig_mode(gpu_id)
            print(enable_mig_out)
            # benchmark on different mig profile instance
            try:
                for mig_profile in mig_profiles:
                    dci_out, dgi_out = reset_mig(gpu_id)
                    print(dci_out)
                    print(dgi_out)
                    cgi_out = create_mig_profile(gpu_id, mig_profile)
                    print(cgi_out)
                    # run benchmark
                    f(*args, **kwargs)
                    print(f"benchmark on {mig_profile} done")
            except Exception as e:
                print(e, f'benchmark failed')
        return inner
    return decorator


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


