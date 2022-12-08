# MPS Experiment

## 0. Setup

### 1. Setup GPU Compute Mode to EXCLUSIVE\_PROCESS (Only do once)

Reqire root:
```shell
nvidia-smi -i 1 -c EXCLUSIVE_PROCESS
```
`-i` specified GPU ID.

Check compute mode:
```shell
nvidia-smi -i 1 -q | grep 'Compute Mode'
```

### 2. Disable MIG on all GPUs (there is a GPU detection bug on PyTorch with MIG)
```shell
nvidia-smi mig -dci
nvidia-smi mig -dgi
nvidia-smi -mig 0
```

### 3. Start MPS

```shell
# export CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps
#export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
nvidia-cuda-mps-control -d
```
If you do not set the NVIDIA MPS Log and Pipe directory, please remeber to enable R/W access for all users:
```shell
chmod 777 /var/log/nvidia-mps
chmod 777 /tmp/nvidia-mps
```

Check the service started
```shell
pidof nvidia-cuda-mps-control
```

### 4. Stop MPS after experiment
```shell
echo quit | nvidia-cuda-mps-control
```

## 1. 2 MPS Instance on A30 with different batch size
```shell
bash mps_2_instance_batch_size.sh
```
