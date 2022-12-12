#! /usr/bin/env bash
GPU_ID=1
# 'resnet152' OOM
MODEL_NAMES=('resnet18' 'resnet34' 'resnet50' 'resnet101')
NUM_TEST_BATCHES=1000
BATCH_SIZE=32

EXP_SAVE_DIR="${PWD}"
cd ../../mig_perf/inference
export PYTHONPATH="${PWD}"

# iterate through batch size list
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Model name ${MODEL_NAME}"
    echo 'Start dummy client 1'
    python client/block_inference.py --dry-run  -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n 1048575  --device-id "${GPU_ID}" -i "${GPU_ID}" > /dev/null 2>&1 &
    CLIENT1_PID=$!

    echo 'Start dummy client 2'
    python client/block_inference.py --dry-run  -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n 1048575  --device-id "${GPU_ID}" -i "${GPU_ID}" > /dev/null 2>&1 &
    CLIENT2_PID=$!

    echo 'Start dummy client 3'
    python client/block_inference.py --dry-run  -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n 1048575  --device-id "${GPU_ID}" -i "${GPU_ID}" > /dev/null 2>&1 &
    CLIENT3_PID=$!

    sleep 5

    echo 'Start profiling client 0'
    python client/block_inference.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}" --device-id "${GPU_ID}" -i "${GPU_ID}" -dbn "${EXP_SAVE_DIR}/model_name_4_instance"

    echo 'Cleaning up...'
    kill -9 $CLIENT1_PID $CLIENT2_PID $CLIENT3_PID

    echo 'Finish!'

    sleep 10
done
