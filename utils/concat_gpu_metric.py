from typing import Union

import pandas as pd
import numpy as np
from pathlib import Path


def timestamps_align(record_file_path: Union[str, Path], dcgm_file_path: Union[str, Path]):
    res = pd.read_csv(record_file_path)
    dcgm = pd.read_csv(dcgm_file_path)
    for index in range(len(res)):
        time_period = [int(res.loc[index, 'start_timestamp']), int(res.loc[index, 'end_timestamp'])]
        gract = []
        fb_used = []
        for i in range(len(dcgm)):
            if int(dcgm.loc[i, 'TimeStamp']) > time_period[1]:
                break
            if time_period[1] >= int(dcgm.loc[i, 'TimeStamp']) >= time_period[0]:
                gract += [float(dcgm.loc[i, 'GRACT'])]
                fb_used += [float(dcgm.loc[i, 'FBUSD'])]
        gract_mean = 0.0 if gract == [] else np.mean(gract)
        gract_std = 0.0 if gract == [] else np.std(gract)
        fb_used_mean = 0.0 if fb_used == [] else np.mean(fb_used)
        fb_used_std = 0.0 if fb_used == [] else np.std(fb_used)
        res.loc[index, 'gract_mean'] = round(gract_mean, 3)
        res.loc[index, 'gract_std'] = round(gract_std, 3)
        res.loc[index, 'fb_used_mean'] = round(fb_used_mean, 3)
        res.loc[index, 'fb_used_std'] = round(fb_used_std, 3)
    save_file = Path(record_file_path).parent / 'integrated_result.csv'
    res.to_csv(save_file)


def main():
    timestamps_align('/data/A100-80g/train/nlp/nlp_train_bsz.csv',
                     '/data/A100-80g/train/nlp\dcgm.csv')

if __name__ == '__main__':
    main()