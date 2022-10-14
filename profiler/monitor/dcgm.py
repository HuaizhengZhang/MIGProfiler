import os
import re
import subprocess
import time
from pathlib import Path
import requests
import pandas as pd


def dcgm_exporter(gpu_i_id, save_dir=None):
    url = "http://127.0.0.1:9400/metrics"
    metric = requests.get(url).text
    timestamp = int(time.time())
    for line in metric.splitlines():
        if line[0]!='#':
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


def dcgm(save_dir, instance_id, logger):
    assert instance_id == 0, "only support instance id 0"
    # TODO: ONLY SUPPORT INSTANCE ID 0
    cmd = f'dcgmi dmon -e 1001,252 -i i:{str(instance_id)}'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    if p.returncode == 0:
        logger.info(f'start dcgm monitor on instance {instance_id} success')
    else:
        logger.warning(
            f'start dcgm monitor on instance {instance_id} failed'
        )
    save_dir = save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = Path(save_dir) / 'dcgm.csv'
    while not p.poll():
        line = p.stdout.readline().split()
        timestamp = int(time.time())
        if 'GPU-I' in line:
            line += [timestamp]
            df = pd.DataFrame([line[2:]], columns=['GRACT', 'FBUSD', 'TimeStamp'])
            if not save_path.exists():
                df.to_csv(save_path, mode='a', header=True, index=False)
            else:
                df.to_csv(save_path, mode='a', header=False, index=False)

if __name__ == "__main__":
    dcgm_exporter(gpu_i_id=1, save_dir="E:\MIGProfiler\profiler\monitor")