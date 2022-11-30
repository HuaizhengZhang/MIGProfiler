import re
import requests

from mig_perf.exporter import DCGM_URL


def dcgm_exporter(gpu_i_id, url=DCGM_URL):
    metric = requests.get(url).text
    if metric is None:
        return None
    if gpu_i_id != 'None':
        for line in metric.splitlines():
            if line[0] != '#':
                gid = re.search(r"GPU_I_ID=\"(.)\"", line).group(1)
                if gid == str(gpu_i_id):
                    profile = re.search(r"GPU_I_PROFILE=\"([0-9]g.[0-9]*gb)\"", line).group(1)
                    if line.split('{')[0] == "DCGM_FI_DEV_FB_USED":
                        fbusd = line.split(' ')[-1]
                    if line.split('{')[0] == "DCGM_FI_PROF_GR_ENGINE_ACTIVE":
                        gract = line.split(' ')[-1]
                    if line.split('{')[0] == "DCGM_FI_DEV_POWER_USAGE":
                        power = line.split(' ')[-1]
        return gract, fbusd, power, profile
    else:
        for line in metric.splitlines():
            if line[0] != '#':
                gid = re.search(r"gpu=\"(.)\"", line).group(1)
                if gid == '0':
                    profile = re.search(r"(NVIDIA [A-Z0-9\-]+)", line).group(1)
                    if line.split('{')[0] == "DCGM_FI_DEV_FB_USED":
                        fbusd = line.split(' ')[-1]
                    if line.split('{')[0] == "DCGM_FI_PROF_GR_ENGINE_ACTIVE":
                        gract = line.split(' ')[-1]
                    if line.split('{')[0] == "DCGM_FI_DEV_POWER_USAGE":
                        power = line.split(' ')[-1]
        return gract, fbusd, power, profile

