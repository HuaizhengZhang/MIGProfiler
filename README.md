# A100_Benchmark
A benchmark script for various deep learning workloads on Nvidia Ampere mig devices.

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
train: True #False for infering, True for training tasks.
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

