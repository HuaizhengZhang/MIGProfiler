# A100_Benchmark
A benchmark script for various deep learning workloads on Nvidia Ampere mig devices.

## Software

DCGM  version: 2.4.6

Mig-parted version 0.5.0

Python 3.9.12

## Quick Start 

### 1. Install

```shell
$ . install.sh
```

### 2. Single instance profiling for a certain workload

#### Configuration

```yaml
model_name: vision_transformer
gpuID: 0
workload: cv_infer
save_dir: /root/A100-benchmark/data/cv_infer/
hydra:
  run:
    dir: /root/MIGProfiler/logs/${now:%Y-%m-%d}/${now:%H-%M-%S}
dcgm:
  save_dir: /root/A100-benchmark/data/cv_infer/

```

#### Run Benchmark

```shell
$ python migprofile_single_instance.py
```

results are saved at `data/single_instance_profile/{model_name}/{train(infer)}/`

### 3. Hybrid workloads profiling for a certain mig partition

#### Configuration

```yaml
partition: [3,4]
gpuID: 0
train_workload: vision_transformer
infer_workload: resnet50
```

#### Run Benchmark

```shell
$ python migprofile_hybrid_workloads.py
```

results are saved at `data/hybrid_workloads_profile/{partition}/`

