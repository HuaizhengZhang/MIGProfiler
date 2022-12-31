#! /usr/bin/env bash
NVIDIA_VISIBLE_DEVICES="0:0,0:1"

docker pull nvcr.io/nvidia/tritonserver:21.09-pyt-python-py3

docker run --gpus="\"device=${NVIDIA_VISIBLE_DEVICES}\"" -d --rm --network=host \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67100864 --name triton \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /"${PWD}":/models/resnet50 \
    nvcr.io/nvidia/tritonserver:21.09-pyt-python-py3 \
    tritonserver --model-repository=/models

sleep 10

# pip install 'tritonclient[http]'
python triton_client.py -m resnet50

docker stop triton