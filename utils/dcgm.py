import os
import re
import time
from pathlib import Path
import requests


def dcgm_exporter(gpu_i_id, save_dir=None):
    url = "http://127.0.0.1:9400/metrics"
    metric = requests.get(url).text
    timestamp = int(time.time())
    for line in metric.splitlines():
        if line[0] != '#':
            gid = re.search(r"GPU_I_ID=\"(.)\"", line).group(1)
            if gid == str(gpu_i_id):
                profile = re.search(r"GPU_I_PROFILE=\"([0-9]g.[0-9]*gb)", line).group(1)
                if line.split('{')[0] == "DCGM_FI_DEV_FB_USED":
                    fbusd = line.split(' ')[-1]
                if line.split('{')[0] == "DCGM_FI_PROF_GR_ENGINE_ACTIVE":
                    gract = line.split(' ')[-1]
    save_file_path = Path(save_dir) / f'dcgm.csv'
    if not Path(save_dir).exists():
        os.makedirs(save_dir)
    if not save_file_path.exists():
        with open(save_file_path, mode='w') as save_file:
            save_file.write("EntityId,Profile,GRACT,FBUSD,TimeStamp\n")

    with open(save_file_path, mode='a') as save_file:
        save_file.write(f"{gid},{profile},{gract},{fbusd},{timestamp}\n")

