import logging
import os
import sys
sys.path.append(os.getcwd())
import subprocess
import time
from pathlib import Path
import hydra
import pandas as pd
from omegaconf import DictConfig
from multiprocessing import Process
from utils.mig_parted import get_mig_devices


@hydra.main(version_base=None, config_path='../configs', config_name='single_instance')
def main(cfg: DictConfig):
    model_name = cfg.model_name
    gpu_id = cfg.gpuID
    workload = cfg.workload
    save_dir = cfg.save_dir
    dcgm_save_dir = cfg.dcgm.save_dir
    logger = logging.getLogger(f"{model_name}_{workload}")
    # enable gpu mig mode
    try:
        logger.info("try to stop dcgm before mig partition")
        stop_dcgm = subprocess.Popen(['systemctl', 'stop', 'dcgm'])
        _ = stop_dcgm.communicate(timeout=5)
        enable_mig = subprocess.Popen(['nvidia-smi', '-i', str(gpu_id), '-mig', '1'])
        _ = enable_mig.communicate(timeout=30)
        time.sleep(2)
        logger.info(f"GPU:{gpu_id} mig mode is enabled")

    except RuntimeError as e:
        logger.error(e, e.__traceback__.tb_lineno, f'enable gpu:{gpu_id} mig mode failed')

    mig_profiles = ['1g.10gb', '2g.20gb', '3g.40gb', '4g.40gb', '7g.80gb']
    # start dcgm
    try:
        start_dcgm = subprocess.Popen(['systemctl', 'start', 'dcgm'])
        _ = start_dcgm.communicate(timeout=5)
        time.sleep(2)
        logger.info(f"dcgm is started")
    except RuntimeError as e:
        logger.error(e, e.__traceback__.tb_lineno, f'start dcgm failed')
    # benchmark on different mig profile instance
    try:
        for mig_profile in mig_profiles:
            try:
                mig_create = subprocess.Popen(['bash', './scripts/mig_controller.sh', str(gpu_id), mig_profile])
                _ = mig_create.communicate(timeout=30)
                time.sleep(2)
                logger.info(f"GPU:{gpu_id} mig profile {mig_profile} is created")
            except RuntimeError as e:
                logger.error(e, "mig partition failed")

            # get mig device uuid
            target_mig_device = get_mig_devices(gpu_id)[0]
            checking_mig_profile, uuid, instance_id = target_mig_device['mig_name'], \
                                                      target_mig_device['uuid'], \
                                                      target_mig_device['instance_id']
            assert checking_mig_profile == mig_profile, f"target mig device does not match, " \
                                                        f"one is {checking_mig_profile}, the other is {mig_profile}"

            # run benchmark
            dcgm_proc = Process(target=dcgm, args=(dcgm_save_dir, instance_id))
            dcgm_proc.daemon = True
            dcgm_proc.start()
            logger.info(f"dcgm process {dcgm_proc.pid} is monitoring")
            time.sleep(2)
            benchmark(workload, uuid, model_name, mig_profile, gpu_id, save_dir)
            logger.info(f"dcgm process {dcgm_proc.pid} is stopped")
            dcgm_proc.terminate()

    except Exception as e:
        logger.exception(e, e.__traceback__.tb_lineno, 'benchmark failed')


def benchmark(workload, cuda_device_uuid, model_name, mig_profile, gpu_id, save_dir):
    try:
        bmk = subprocess.Popen(
            ['bash', f'./scripts/{workload}.sh', f'{cuda_device_uuid}', f'{model_name}', f'{mig_profile}', f'{gpu_id}', f'{save_dir}']
        )
        _ = bmk.communicate()
        time.sleep(2)
    except Exception as e:
        print("benchmark script failed: {}".format(e))
        bmk.terminate()


def dcgm(save_dir, instance_id):
    try:
        # TODO: ONLY SUPPORT INSTANCE ID 0
        dcgm = subprocess.Popen(
            ['dcgmi', 'dmon', '-e', '1001,252', '-i', f'i:0'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8")
        save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = Path(save_dir) / 'dcgm.csv'
        while not dcgm.poll():
            line = dcgm.stdout.readline().split()
            timestamp = int(time.time())
            if 'GPU-I' in line:
                line += [timestamp]
                df = pd.DataFrame([line[2:]], columns=['GRACT', 'FBUSD', 'TimeStamp'])
                if not save_path.exists():
                    df.to_csv(save_path, mode='a', header=True, index=False)
                else:
                    df.to_csv(save_path, mode='a', header=False, index=False)
    except RuntimeError as e:
        print("dcgm failed: {}".format(e))
        dcgm.terminate()


if __name__ == '__main__':
    main()