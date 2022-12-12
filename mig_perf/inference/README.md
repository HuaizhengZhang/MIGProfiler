# Lightweight PyTorch Inference Server

## Install

Requirements:
- PyTorch with CUDA
- OpenCV
- Sanic

```shell
# create virtual environment
conda create -n mig-perf python=3.8
conda activate mig-perf

# install required packages
conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c conda-forge opencv
pip install sanic tqdm prometheus_client
```

## Start the inference service
Switch to the correct Python Environment:
```shell
conda activate mig-perf
```

### 1. Start server:
```shell
export PYTHONPATH="${PWD}"
MODEL_NAME="resnet18" TASK="image_classification" DEVICE_ID="0" python server/app.py
```

With MIG enabled, use UUID for the DEVICE_ID:
```shell
MODEL_NAME="resnet18" TASK="image_classification" DEVICE_ID="MIG-cea7b568-2767-5e23-8d99-3d4512238e6f" python server/app.py
```

### 2. Start DCGM GPU monitoring service
```shell
# execute the command at mig_perf/inference directory
docker run -d --rm --gpus all --net mig_perf -p 9400:9400  \
  -v "${PWD}/client/dcp-metrics-included.csv:/etc/dcgm-exporter/customized.csv" \
  --name dcgm_exporter --cap-add SYS_ADMIN   nvcr.io/nvidia/k8s/dcgm-exporter:2.4.7-2.6.11-ubuntu20.04 \
  -c 500 -f /etc/dcgm-exporter/customized.csv -d f
```
Refer [NVIDIA DCGM website](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/dcgm-exporter.html#dcgm-exporter-customization) for the usage.

## Test the inference service
We test the inference service by script shown below. It sends a burst of requests subjects to a Poisson distribution with a specific arrival rate.
```shell
# remember to export PYTHONPATH at the project root
export PYTHONPATH="${PWD}"
python client/pytorch_cv_client.py -r 20 -b 1 -t 30 -P -m resnet18
```
The test script performs a 30-second test with request arrival rate (`-r`) = 20 req/sec, with a batch size (`-b`) = 1.  

## Stop the system

### 1. Stop the server
Press Ctrl + C to stop the server.

### 2. Stop the DCGM GPU monitering service
```shell
docker stop dcgm_exporter
```

## Usage

### Inference Server Usage
Set the following environment variable to config the server:

| Env Var Name         | Required | Description                                                                                   |
|----------------------|----------|-----------------------------------------------------------------------------------------------|
| MODEL_NAME           | YES      | Model name to be loaded.                                                                      |
| TASK                 | YES      | Task name for the inference service. One of 'image_classification', 'sequence_classification' |
 | DEVICE_ID            | YES      | GPU ID / GPU UUID                                                                             |
 | PORT                 | NO       | Server listening port number. Default to 50075.                                               |
 | SERVER_PREPROCESSING | NO       | Pre-process request on the server side. Default to False                                      |

### Client Usage
TODO
