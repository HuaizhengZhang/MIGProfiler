from typing import Union

import pandas as pd
import numpy as np
from pathlib import Path


def timestamps_align(record_file_path: Union[str, Path], dcgm_file_path: Union[str, Path]):
    infer_res = pd.read_csv(record_file_path)
    dcgm_res = pd.read_csv(dcgm_file_path)
    for index in range(len(infer_res)):
        time_period = [int(infer_res.loc[index, 'start_timestamp']), int(infer_res.loc[index, 'end_timestamp'])]
        gract = []
        fb_used = []
        for i in range(len(dcgm_res)):
            if int(dcgm_res.loc[i, 'TimeStamp']) > time_period[1]:
                break
            if time_period[1] >= int(dcgm_res.loc[i, 'TimeStamp']) >= time_period[0]:
                gract += [float(dcgm_res.loc[i, 'GRACT'])]
                fb_used += [float(dcgm_res.loc[i, 'FBUSD'])]
        gract_mean = 0.0 if gract == [] else np.mean(gract)
        gract_std = 0.0 if gract == [] else np.std(gract)
        fb_used_mean = 0.0 if fb_used == [] else np.mean(fb_used)
        fb_used_std = 0.0 if fb_used == [] else np.std(fb_used)
        infer_res.loc[index, 'gract_mean'] = round(gract_mean, 3)
        infer_res.loc[index, 'gract_std'] = round(gract_std, 3)
        infer_res.loc[index, 'fb_used_mean'] = round(fb_used_mean, 3)
        infer_res.loc[index, 'fb_used_std'] = round(fb_used_std, 3)
    return infer_res


def gather_from_profiles(results_dir: Union[str, Path], dcgm_file_path: Union[str, Path]):
    result_of_profiles = []
    model_name = str(results_dir).split('/')[-1].split('\\')[-1]
    assert Path(results_dir).exists() and Path(dcgm_file_path).exists()
    for file in Path(results_dir).rglob('*MIG*'):
        result_of_profiles.append(timestamps_align(file, dcgm_file_path))
    gathered = pd.concat(result_of_profiles)
    gathered.to_csv(str(results_dir) + '/' + model_name + '.csv')


def main():
    gather_from_profiles('E:/InferFinetuneBenchmark/data/pretrain_resnet50_moco',
                         'E:/InferFinetuneBenchmark/data/pretrain_resnet50_moco/dcgm.csv')

if __name__ == '__main__':
    main()