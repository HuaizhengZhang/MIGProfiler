# MIG Profiler
A dockerized pipeline for NVIDIA MIG devices deep learning  workload profiling.

## Software

DCGM  version: 2.4.6

Python 3.9.12

## Quick Start 

## 1. Install 

**git install**

```bash
$ git clone https://github.com/MLSysOps/MIGProfiler.git
$ cd MIGProfiler
$ . install.sh
```

##### Check installation

```bash
$ docker images
```

output:

```
REPOSITORY                         TAG                           IMAGE ID       CREATED          SIZE
mig-perf/profiler                  1.0                           e42bff41025d   31 minutes ago   6.25GB
nvcr.io/nvidia/k8s/dcgm-exporter   2.4.7-2.6.11-ubuntu20.04      f61f58af30cd   3 weeks ago      953MB
```



## 2. Run profiling container on a MIG device 

Make sure that **no cuda process** is running on the GPU you are going to test.

1. enable MIG mode for `GPU 0`

   ```shell
   $ nvidia-smi -i 0 -mig 1
   ```

2. get possible mig devices for `GPU 0` 

   ```shell
   $ nvidia-smi mig -i 0 -lgip
   ```

   output:

   ```
   +-----------------------------------------------------------------------------+
   | GPU instance profiles:                                                      |
   | GPU   Name             ID    Instances   Memory     P2P    SM    DEC   ENC  |
   |                              Free/Total   GiB              CE    JPEG  OFA  |
   |=============================================================================|
   |   0  MIG 1g.10gb       19     4/7        9.50       No     14     0     0   |
   |                                                             1     0     0   |
   +-----------------------------------------------------------------------------+
   |   0  MIG 1g.10gb+me    20     1/1        9.50       No     14     1     0   |
   |                                                             1     1     1   |
   +-----------------------------------------------------------------------------+
   |   0  MIG 2g.20gb       14     2/3        19.50      No     28     1     0   |
   |                                                             2     0     0   |
   +-----------------------------------------------------------------------------+
   |   0  MIG 3g.40gb        9     1/2        39.50      No     42     2     0   |
   |                                                             3     0     0   |
   +-----------------------------------------------------------------------------+
   |   0  MIG 4g.40gb        5     0/1        39.50      No     56     2     0   |
   |                                                             4     0     0   |
   +-----------------------------------------------------------------------------+
   |   0  MIG 7g.80gb        0     0/1        79.25      No     98     5     0   |
   |                                                             7     1     1   |
   +-----------------------------------------------------------------------------+
   
   ```

   

3. set up the mig device configuration you want to profile, for example, here we will profile on `MIG 4g.40gb`

   ```shell
   $ nvidia-smi mig -i 0 -cgi MIG 4g.40gb -C
   ```

   output:

   

4.  

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
