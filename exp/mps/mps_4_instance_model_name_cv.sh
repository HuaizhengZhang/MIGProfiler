#! /usr/bin/env bash
GPU_ID=1
# 'resnet152' OOM
MODEL_NAMES=('resnet18' 'resnet34' 'resnet50' 'resnet101')
NUM_TEST_BATCHES=1000
BATCH_SIZE=32

EXP_SAVE_DIR="${PWD}"
cd ../../mig_perf/inference
export PYTHONPATH="${PWD}"

echo 'Enable MPS'
nvidia-cuda-mps-control -d
echo "MPS control running at $(pidof nvidia-cuda-mps-control)"

echo 'Start DCGM'
docker run -d --rm --gpus all --net mig_perf -p 9400:9400  \
  -v "${EXP_SAVE_DIR}/../../mig_perf/inference/client/dcp-metrics-included.csv:/etc/dcgm-exporter/customized.csv" \
  --name dcgm_exporter --cap-add SYS_ADMIN   nvcr.io/nvidia/k8s/dcgm-exporter:2.4.7-2.6.11-ubuntu20.04 \
  -c 500 -f /etc/dcgm-exporter/customized.csv -d f
sleep 3
docker ps

# iterate through batch size list
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Model name ${MODEL_NAME}"
    echo 'Start dummy client 1'
    python client/block_inference_cv.py --dry-run  -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n 1048575 -i "${GPU_ID}" > /dev/null 2>&1 &
    CLIENT1_PID=$!

    echo 'Start dummy client 2'
    python client/block_inference_cv.py --dry-run  -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n 1048575 -i "${GPU_ID}" > /dev/null 2>&1 &
    CLIENT2_PID=$!

    echo 'Start dummy client 3'
    python client/block_inference_cv.py --dry-run  -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n 1048575 -i "${GPU_ID}" > /dev/null 2>&1 &
    CLIENT3_PID=$!

    sleep 5

    echo 'Start profiling client 0'
    python client/block_inference_cv.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}" -i "${GPU_ID}" -dbn "${EXP_SAVE_DIR}/model_name_4_instance"

    echo 'Cleaning up...'
    kill -9 $CLIENT1_PID $CLIENT2_PID $CLIENT3_PID

    echo 'Finish!'

    sleep 10
done

echo 'Disable MPS'
echo quit | nvidia-cuda-mps-control

echo 'Shutdown DCGM'
docker stop dcgm_exporter
