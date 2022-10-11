import logging
import os
import sys
from controller.utils import get_mig_devices, enable_mig_mode, reset_mig, create_mig_profile
from monitor.dcgm import dcgm_monitor, stop_dcgm
import subprocess
import hydra
from omegaconf import DictConfig
sys.path.append(os.getcwd())


@hydra.main(version_base=None, config_path='../configs', config_name='single_instance')
def main(cfg: DictConfig):
    model_name = cfg.model_name
    gpu_id = cfg.gpuID
    workload = cfg.workload
    save_dir = cfg.save_dir
    dcgm_save_dir = cfg.dcgm.save_dir
    mig_profiles = cfg.mig_profiles
    logger = logging.getLogger(f"{model_name}_{workload}")

    @dcgm_monitor(save_dir=dcgm_save_dir, instance_id=0, logger=logger)
    def benchmark(workload, cuda_device_uuid, model_name, mig_profile, gpu_id, save_dir):
        cmd = ['bash', f'./scripts/{workload}.sh', f'{cuda_device_uuid}', f'{model_name}', f'{mig_profile}',
                 f'{gpu_id}', f'{save_dir}']
        try:
            bmk = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            logger.info(bmk.communicate()[0].decode('utf_8'))
        except Exception as e:
            logger.warning(f"benchmark script for {model_name}:{workload} failed on {cuda_device_uuid}, {mig_profile},"
                           f" debug info: {e}")
            bmk.terminate()

    # stop dcgm service
    dcgm_stop_out = stop_dcgm()
    logger.info(dcgm_stop_out) if dcgm_stop_out is None else logger.info("dcgm is already stopped")

    # enable gpu mig mode
    enable_mig_out = enable_mig_mode(gpu_id)
    logger.info(enable_mig_out)

    # benchmark on different mig profile instance
    try:
        for mig_profile in mig_profiles:
            dci_out, dgi_out = reset_mig(gpu_id)
            logger.info(dci_out)
            logger.info(dgi_out)
            cgi_out = create_mig_profile(gpu_id, mig_profile)
            logger.info(cgi_out)
            # get mig device uuid
            target_mig_device = get_mig_devices(gpu_id)[0]
            checking_mig_profile, uuid, instance_id = target_mig_device['mig_name'], \
                                                      target_mig_device['uuid'], \
                                                      target_mig_device['instance_id']
            assert checking_mig_profile == mig_profile, f"target mig device does not match, " \
                                                        f"one is {checking_mig_profile}, the other is {mig_profile}"
            # @yizheng here the benchmark need run in docker
            # run benchmark
            benchmark(workload, uuid, model_name, mig_profile, gpu_id, save_dir)

    except Exception as e:
        logger.exception(e, f'benchmark failed')


if __name__ == '__main__':
    main()