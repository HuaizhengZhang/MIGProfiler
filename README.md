# A100_Benchmark
A benchmark script for various deep learning workloads on Nvidia Ampere mig devices.

## Software

DCGM  version: 2.4.6

Mig-parted version 0.5.0

Python 3.9.12

## Quick Start 

## 1. Install DCGM

### Ubuntu LTS

**Set up the CUDA network repository meta-data, GPG key:**

```bash
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
$ sudo dpkg -i cuda-keyring_1.0-1_all.deb
$ sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
```

**Install DCGM**

```bash
$ sudo apt-get update \
    && sudo apt-get install -y datacenter-gpu-manager
```

### Red Hat

**Set up the CUDA network repository meta-data, GPG key:**

```bash
$ sudo dnf config-manager \
    --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
```

**Install DCGM**

```bash
$ sudo dnf clean expire-cache \
    && sudo dnf install -y datacenter-gpu-manager
```

### Check installation

```bash
$ sudo systemctl --now enable nvidia-dcgm
```

## 2. Prepare python environment

```shell
$ conda env create -f environments.yaml
$ conda activate benchmark
$ pip install requirements.txt
```

## 3. Single instance profiling for a certain workload

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
$ cd /Path_to/MIGProfiler/
$ python migprofile_single_instance.py
```

results are saved at `save_dir`

## 4. Visualize results

We have visualized some results to look into the benchmark. You can refer to /doc/notebook/plot_results.ipynb to draw pcitures for your own data. Here are visualization results of profiling with seving a ViT model on NVIDIA A100. 


|FB Used|Graphics Engine Activity|Avg. Latency (ms)|Throughput (request/s)|
|:--:|:--:|:--:|:--:|
|![](https://github.com/MLSysOps/MIGProfiler/blob/0cb58f51c557dde9f494acabdd903d5432c946b1/data/A100-80g/infer/cv/vision_transformer_fbusd_bsz_compare.svg)|![](https://github.com/MLSysOps/MIGProfiler/blob/0cb58f51c557dde9f494acabdd903d5432c946b1/data/A100-80g/infer/cv/vision_transformer_gract_bsz_compare.svg)|![](https://github.com/MLSysOps/MIGProfiler/blob/0cb58f51c557dde9f494acabdd903d5432c946b1/data/A100-80g/infer/cv/vision_transformer_latency_bsz_compare.svg)|![](https://github.com/MLSysOps/MIGProfiler/blob/0cb58f51c557dde9f494acabdd903d5432c946b1/data/A100-80g/infer/cv/vision_transformer_throughput_bsz_compare.svg)|
