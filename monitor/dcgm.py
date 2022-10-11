import os
import subprocess
import time
from multiprocessing import Process
from pathlib import Path
import pandas as pd
from functools import wraps


def dcgm_monitor(save_dir, instance_id, logger):
    def dcgm_decorater(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            # start dcgm
            start_dcgm()
            dcgm_proc = Process(target=dcgm, args=(save_dir, instance_id, logger))
            dcgm_proc.daemon = True
            dcgm_proc.start()
            logger.info(f"dcgm process {dcgm_proc.pid} is monitoring")
            time.sleep(2)
            f(*args, **kwargs)
            logger.info(f"dcgm process {dcgm_proc.pid} is stopped")
            dcgm_proc.terminate()

        return decorated

    return dcgm_decorater


def start_dcgm():
    cmd = 'systemctl start dcgm'
    # start dcgm
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.communicate()[0].decode('utf-8')


def stop_dcgm():
    cmd = 'systemctl stop dcgm'
    # start dcgm
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.communicate()[0].decode('utf-8')


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
