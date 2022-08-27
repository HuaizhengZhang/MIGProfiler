import os
import subprocess
from pathlib import Path

import hydra
import pandas as pd
import time
from omegaconf import DictConfig


@hydra.main(config_path='configs', config_name='dcgm_recorder', version_base=None)
def main(cfg: DictConfig):
    try:
        dcgm = subprocess.Popen(
            ['dcgmi', 'dmon', '-e', '1001,252', '-g', str(cfg.group_id)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8")
        cnt = 0
        first_row = True
        save_dir = cfg.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = Path(save_dir) / 'dcgm.csv'
        while not dcgm.poll():
            line = dcgm.stdout.readline().split()
            timestamp = int(time.time())
            if '#' in line:
                cnt = cnt + 1
            if 'GPU-I' in line and int(line[1]) == cfg.instance_id:
                line += [timestamp]
                df = pd.DataFrame([line[1:]], columns=['EntityId', 'GRACT', 'FBUSD', 'TimeStamp'])
                if first_row:
                    df.to_csv(save_path, mode='a', header=True, index=False)
                    first_row = False
                else:
                    df.to_csv(save_path, mode='a', header=False, index=False)
    except Exception as e:
        print("dcgm failed: {}".format(e))
        dcgm.terminate()


if __name__ == '__main__':
    main()