#! /usr/bin/env bash
NVIDIA_VISIBLE_DEVICES="0:0,0:1"

docker pull tensorflow/serving:latest-gpu

docker run --gpus "\"device=${NVIDIA_VISIBLE_DEVICES}\"" -d --rm -p 8501:8501 --name tfs \
    -v "${PWD}:/models/resnet50" \
    -e MODEL_NAME=resnet50 \
    tensorflow/serving:latest-gpu


sleep 10

python restful_client.py -m resnet50

docker stop tfs
