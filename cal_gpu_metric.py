import pandas as pd
import numpy as np

infer_res = pd.read_csv("data/infer_seq/mpnet/paraphrase-multilingual-mpnet-base-v2_MIG_profile_19.csv")
dcgm_res = pd.read_csv("data/infer_seq/dcgm.csv")
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
infer_res.to_csv("data/infer_seq/mpnet/mpnet_infer_05g_seq.csv")
