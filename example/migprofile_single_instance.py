import logging
import os
import sys
import docker

from mig_perf.controller.utils import enable_mig_mode, reset_mig, create_mig_profile

profiler = docker.APIClient(base_url='tcp://127.0.0.1:9709')
dcgm = docker.APIClient(base_url='tcp://127.0.0.1:9400')

sys.path.append(os.getcwd())


def main():
    gpu_id = 0
    mig_profiles = ['1g.10gb', '2g.20gb', '3g.40gb', '4g.40gb', '7g.80gb']
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
            # @yizheng here the benchmark need run in docker
            # run benchmark
            """
            here we should communicate with container and run python profiler.py in docker
            """
            result = profile()

    except Exception as e:
        logger.exception(e, f'benchmark failed')


def profile():
    return None


if __name__ == '__main__':
    main()