import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from benchmark import cv_train, nlp_infer, nlp_train, cv_infer
pd.set_option('display.max_columns', None)
PLACES365_DATA_DIR = "places365/"
RESULT_DATA_DIR = "data"

"""
 docker run -d --rm --gpus all --net mig_perf -p 9400:9400  \
--name dcgm_exporter --cap-add SYS_ADMIN   nvcr.io/nvidia/k8s/dcgm-exporter:2.4.7-2.6.11-ubuntu20.04 \
-c 100 -f /etc/dcgm-exporter/dcp-metrics-included.csv -d i

docker run --rm --gpus 'device=0:0' --net mig_perf \
--name profiler --cap-add SYS_ADMIN --shm-size="15g" \
-v /root/places365_standard/:/workspace/places365/  \
-v /root/MIGProfiler/data/:/workspace/data/ \
-v /root/MIGProfiler/logs/:/workspace/logs/  \
mig-perf/profiler:1.0 "model_name=vision_transformer" "workload=cv_infer"
"""


@hydra.main(version_base=None, config_path='../../configs', config_name='single_instance')
def run(cfgs: DictConfig):
    model_name = cfgs.model_name
    workload = cfgs.workload
    default_batchsize = cfgs.benchmark_plan.default_batchsize
    default_seqlenth = cfgs.benchmark_plan.default_seqlenth
    batch_sizes = cfgs.benchmark_plan.batch_sizes
    seq_lengths = cfgs.benchmark_plan.sequence_lengths
    fixed_time = cfgs.benchmark_plan.fixed_time
    logger = logging.getLogger(f"{model_name}_{workload}")
    ret = None

    if workload == 'cv_infer':
        result = []
        for batch_size in batch_sizes:
            try:
                cv_infer_res = cv_infer(model_name, fixed_time, batch_size, PLACES365_DATA_DIR)
                logger.info(f"batch size={batch_size} inference finished, result:\n{cv_infer_res}")
                result.append(cv_infer_res)
            except Exception as e:
                logger.exception(e)
        ret = pd.concat(result)
    if workload == 'cv_train':
        result = []
        for batch_size in batch_sizes:
            try:
                cv_train_res = cv_train(model_name, fixed_time, batch_size, PLACES365_DATA_DIR)
                result.append(cv_train_res)
                logger.info(f"batch size={batch_size} training finished, result:\n{cv_train_res}")
            except Exception as e:
                logger.exception(e)
        ret = pd.concat(result)
    if workload == 'nlp_infer':
        result_seq = []
        for seq_length in seq_lengths:
            try:
                nlp_infer_seq_res = nlp_infer(model_name, fixed_time, batch_size=default_batchsize, seq_length=seq_length)
                logger.info(f"batch size={default_batchsize} sequence length={seq_length} inference finished, result:\n{nlp_infer_seq_res}")
                result_seq.append(nlp_infer_seq_res)
            except Exception as e:
                logger.exception(e)
        ret_seq = pd.concat(result_seq)
        result_bsz = []
        for batch_size in batch_sizes:
            try:
                nlp_infer_bsz_res = nlp_infer(model_name, fixed_time, batch_size=batch_size, seq_length=default_seqlenth)
                logger.info(
                    f"batch size={batch_size} sequence length={default_seqlenth} inference finished, result:\n{nlp_infer_bsz_res}")
                result_seq.append(nlp_infer_bsz_res)
            except Exception as e:
                logger.exception(e)
        ret_bsz = pd.concat(result_bsz)
        ret = pd.concat([ret_seq, ret_bsz])
    if workload == 'nlp_train':
        result_seq = []
        for seq_length in seq_lengths:
            try:
                nlp_train_seq_res = nlp_train(model_name, fixed_time, batch_size=default_batchsize,
                                              seq_length=seq_length)
                logger.info(
                    f"batch size={default_batchsize} sequence length={seq_length} training finished, result:\n{nlp_train_seq_res}")
                result_seq.append(nlp_train_seq_res)
            except Exception as e:
                logger.exception(e)
        ret_seq = pd.concat(result_seq)
        result_bsz = []
        for batch_size in batch_sizes:
            try:
                nlp_train_bsz_res = nlp_train(model_name, fixed_time, batch_size=batch_size,
                                              seq_length=default_seqlenth)
                logger.info(
                    f"batch size={batch_size} sequence length={default_seqlenth} training finished, result:\n{nlp_train_bsz_res}")
                result_seq.append(nlp_train_bsz_res)
            except Exception as e:
                logger.exception(e)
        ret_bsz = pd.concat(result_bsz)
        ret = pd.concat([ret_seq, ret_bsz])
    # mount local volumn ./workspace
    save_file = Path(f'{RESULT_DATA_DIR}/{model_name}_{workload}.csv')
    if save_file.exists():
        ret.to_csv(f'{RESULT_DATA_DIR}/{model_name}_{workload}.csv', mode='a', header=False)
    else:
        ret.to_csv(f'{RESULT_DATA_DIR}/{model_name}_{workload}.csv', mode='a', header=True)
    logger.info(f"final result: \n {ret}")


if __name__ == '__main__':
    run()


