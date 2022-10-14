import logging
import os
import sys
from controller.utils import get_mig_devices, enable_mig_mode, reset_mig, create_mig_profile
import hydra
from omegaconf import DictConfig

sys.path.append(os.getcwd())


@hydra.main(version_base=None, config_path='../configs', config_name='controller')
def main(cfg: DictConfig):
    gpu_id = cfg.gpu_id
    mig_profiles = cfg.mig_profiles
    logger = logging.getLogger(f"single_instance_benchmark")

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
            """
            here we should communicate with container and run python profiler.py in docker
            """
            profile()

    except Exception as e:
        logger.exception(e, f'benchmark failed')


def profile():
    pass


if __name__ == '__main__':
    main()