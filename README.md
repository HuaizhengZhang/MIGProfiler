# MIG Profiler
A dockerized toolkit for Nvidia [MIG](https://www.nvidia.com/en-sg/technologies/multi-instance-gpu/) GPU (Ampere Series) profiling with deep learning workloads.


## Quick Start 

## 1. Installation 

You can install the toolkit by Git and Github, make sure you have [Docker](https://www.docker.com/) installed on your device. 

```bash
$ git clone https://github.com/MLSysOps/MIGProfiler.git
$ cd MIGProfiler
$ . install.sh
```

The installation script will help you fetch several Docker images, for example

``` bash
REPOSITORY                         TAG                           IMAGE ID       CREATED          SIZE
mig-perf/profiler                  1.0                           e42bff41025d   31 minutes ago   6.25GB
nvcr.io/nvidia/k8s/dcgm-exporter   2.4.7-2.6.11-ubuntu20.04      f61f58af30cd   3 weeks ago      953MB
```


## 2. Profiling Deep Learning Workloads

Make sure that **no cuda process** is running on the GPU you are going to test.

### 1. Enable MIG mode

Choose a gpu that supports MIG on your machine for testing, here we use `GPU 0`,

   ```shell
   $ nvidia-smi -i 0 -mig 1
   ```

### 2. Get possible mig devices

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

   

### 3. Set up the MIG device configuration

You can set up the MIG configuration you would like to profile. For example, here we will profile on `MIG 4g.40gb` configuration.

   ```shell
   $ nvidia-smi mig -i 0 -cgi 4g.40gb -C
   ```

   output:

   ```
   Successfully created GPU instance ID  2 on GPU  0 using profile MIG 4g.40gb (ID  5)
   Successfully created compute instance ID  0 on GPU  0 GPU instance ID  2 using profile MIG 4g.40gb (ID  3)
   ```

### 4. Acquire MIG device IDs

   ```shell
   $ nvidia-smi -L && nvidia-smi mig -lci
   ```

   output:

   ```
   GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-ddb16622-80d1-5e78-5bf7-a0ecfd723d0a)
     MIG 4g.40gb     Device  0: (UUID: MIG-fca28630-a1bd-5e73-9206-ccd17a013f5f)
   GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-4ce12868-fd2c-46a5-1451-877de5670d53)
   GPU 2: NVIDIA A100-SXM4-80GB (UUID: GPU-b9ce234f-fb2b-32e8-db8f-e3bc7f7306e4)
   GPU 3: NVIDIA A100-SXM4-80GB (UUID: GPU-db8e9cfe-637e-6b4a-c689-0da95fac5a60)
   GPU 4: NVIDIA A100-SXM4-80GB (UUID: GPU-2b558483-cff1-c170-fcdb-846979b5faa5)
   GPU 5: NVIDIA A100-SXM4-80GB (UUID: GPU-7bca71eb-da0f-8476-3988-a66bb2c84d86)
   GPU 6: NVIDIA A100-SXM4-80GB (UUID: GPU-5f50ff74-a34c-9487-d21d-c5fa409fc74d)
   GPU 7: NVIDIA A100-SXM4-80GB (UUID: GPU-4a12fe94-db8e-3e7c-5a56-33ef4ad777e3)
   +--------------------------------------------------------------------+
   | Compute instances:                                                 |
   | GPU     GPU       Name             Profile   Instance   Placement  |
   |       Instance                       ID        ID       Start:Size |
   |         ID                                                         |
   |====================================================================|
   |   0      2       MIG 4g.40gb          3         0          0:4     |
   +--------------------------------------------------------------------+
   
   ```

   Here we get the `created 4g.40gb ` has `device_id=0`,  `gpu_instance_id=2 `.

### 5.  Sending profiling workloads

   ```shell
   # start dcgm-exporter
   $ docker run -d --rm --gpus all --net mig_perf -p 9400:9400  \
   --name dcgm_exporter --cap-add SYS_ADMIN   nvcr.io/nvidia/k8s/dcgm-exporter:2.4.7-2.6.11-ubuntu20.04 \
   -c 100 -f /etc/dcgm-exporter/dcp-metrics-included.csv -d i
   # start profiling
   $ docker run --rm --gpus 'device=0:0' --net mig_perf \
   --name profiler --cap-add SYS_ADMIN --shm-size="15g" \
   -v /root/MIGProfiler/data/:/workspace/data/  \
   mig-perf/profiler:1.0 "model_name=vision_transformer" "workload=cv_infer" "gpu_i_id=2"
   # stop dcgm-exporter
   $ docker stop dcgm_exporter
   ```

**Arguments Clarification:**

  1. dcgm-exporter: we do not recommend you to change the arguments of dcgm-exporter container.

  2. profiler:  
     - `--gpus`: use format as "device={`gpu_id`}:{`device_id`}" to provide the target mig device to profiler container,

     - `--shm-size`: shared memory for profiler container, we recommend that it should be larger than 4g.

     - `-v`: mounting data for container. use format as `"path/to/MIGProfiler/data/:/workspace/data/"`.

     - `model_name`: currently 'vision_transformer', 'resnet50', 'swin_transformer' is supported

     - `workload`: according to your `model_name` and tasks(train or infer), four workloads are supported: ' cv_infer', 'cv_train', 'nlp_infer', 'nlp_train'.

     - `gpu_i_id`: GPU Instance ID you get in `step 4`, this is for dcgm-exporter.

     results will be saved at `path/to/MIGProfiler/data/`.

## 3. Visualize Results

We have visualized some results to look into the benchmark. You can refer to `/doc/notebook/plot_results.ipynb` to draw pcitures for your own data. Here are visualization results of profiling with seving a ViT model on NVIDIA A100. 


|FB Used|Graphics Engine Activity|Avg. Latency (ms)|Throughput (request/s)|
|:--:|:--:|:--:|:--:|
|![](https://github.com/MLSysOps/MIGProfiler/blob/0cb58f51c557dde9f494acabdd903d5432c946b1/data/A100-80g/infer/cv/vision_transformer_fbusd_bsz_compare.svg)|![](https://github.com/MLSysOps/MIGProfiler/blob/0cb58f51c557dde9f494acabdd903d5432c946b1/data/A100-80g/infer/cv/vision_transformer_gract_bsz_compare.svg)|![](https://github.com/MLSysOps/MIGProfiler/blob/0cb58f51c557dde9f494acabdd903d5432c946b1/data/A100-80g/infer/cv/vision_transformer_latency_bsz_compare.svg)|![](https://github.com/MLSysOps/MIGProfiler/blob/0cb58f51c557dde9f494acabdd903d5432c946b1/data/A100-80g/infer/cv/vision_transformer_throughput_bsz_compare.svg)|
