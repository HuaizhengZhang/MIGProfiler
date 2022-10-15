import logging
import subprocess
import time
from multiprocessing import Process
import hydra
from utils import dcgm


@hydra.main(version_base=None, config_path='../configs', config_name='single_instance')
def benchmark(workload, cuda_device_uuid, model_name, mig_profile, gpu_id, instance_id, save_dir):
    logger = logging.getLogger(f"{model_name}_{workload}")
    cmd = ['bash', f'./scripts/{workload}.sh', f'{cuda_device_uuid}', f'{model_name}', f'{mig_profile}',
           f'{gpu_id}', f'{save_dir}']
    try:
        benchmark_proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        logger.info(benchmark_proc.communicate()[0].decode('utf_8'))
    except Exception as e:
        logger.warning(f"benchmark script for {model_name}:{workload} failed on {cuda_device_uuid}, {mig_profile},"
                       f" debug info: {e}")
        benchmark.terminate()


if __name__ == '__main__':
    benchmark()


