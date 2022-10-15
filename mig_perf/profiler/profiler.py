import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from benchmark import cv_train, nlp_infer, nlp_train, cv_infer


@hydra.main(version_base=None, config_path='../../example/configs', config_name='single_instance')
def run(cfgs: DictConfig):
    model_name = cfgs.model_name
    workload = cfgs.workload
    default_batchsize = cfgs.default_batchsize
    default_seqlenth = cfgs.default_seqlenth
    batch_sizes = cfgs.benchmark_plan.batch_sizes
    seq_lengths = cfgs.benchmark_plan.sequence_lengths
    fixed_time = cfgs.benchmark_plan.fixed_time
    logger = logging.getLogger(f"{model_name}_{workload}")
    ret = None

    if workload == 'cv_infer':
        result = []
        for batch_size in batch_sizes:
            try:
                result.append(cv_infer(model_name, fixed_time, batch_size))
            except Exception as e:
                logger.exception(e)
        ret = pd.concat(result)
    if workload == 'cv_train':
        result = []
        for batch_size in batch_sizes:
            try:
                result = cv_train(model_name, fixed_time, batch_size)
            except Exception as e:
                logger.exception(e)
        ret = pd.concat(result)
    if workload == 'nlp_infer':
        result_seq = []
        for seq_length in seq_lengths:
            try:
                result_seq = nlp_infer(model_name, fixed_time, batch_size=default_batchsize, seq_length=seq_length)
            except Exception as e:
                logger.exception(e)
        ret_seq = pd.concat(result_seq)
        result_bsz = []
        for batch_size in batch_sizes:
            try:
                result_bsz = nlp_infer(model_name, fixed_time, batch_size=batch_size, seq_length=default_seqlenth)
            except Exception as e:
                logger.exception(e)
        ret_bsz = pd.concat(result_bsz)
        ret = pd.concat([ret_seq, ret_bsz])
    if workload == 'nlp_train':
        result_seq = []
        for seq_length in seq_lengths:
            try:
                result_seq = nlp_train(model_name, fixed_time, batch_size=default_batchsize, seq_length=seq_length)
            except Exception as e:
                logger.exception(e)
        ret_seq = pd.concat(result_seq)
        result_bsz = []
        for batch_size in batch_sizes:
            try:
                result_bsz = nlp_train(model_name, fixed_time, batch_size=batch_size, seq_length=default_seqlenth)
            except Exception as e:
                logger.exception(e)
        ret_bsz = pd.concat(result_bsz)
        ret = pd.concat([ret_seq, ret_bsz])
    # mount local volumn ./workspace
    save_file = Path(f'./workspace/{model_name}_{workload}.csv')
    if save_file.exists():
        ret.to_csv(f'./workspace/{model_name}_{workload}.csv', mode='a', header=False)
    else:
        ret.to_csv(f'./workspace/{model_name}_{workload}.csv', mode='a', header=True)
    logger.info(f"result: {ret}")


if __name__ == '__main__':
    run()


