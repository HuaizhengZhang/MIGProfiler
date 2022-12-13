#! /usr/bin/env bash
GPU_ID=1
MODEL_NAME=resnet18
NUM_TEST_BATCHES=1000
BATCH_SIZES=(1 2 4 8 16 32 64)

EXP_SAVE_DIR="${PWD}"
cd ../../mig_perf/inference
export PYTHONPATH="${PWD}"

# iterate through batch size list
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    echo "Batch size ${BATCH_SIZE}"
    echo 'Start dummy client 1'
    python client/block_inference_cv.py --dry-run  -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n 1048575  --device-id "${GPU_ID}" -i "${GPU_ID}" > /dev/null 2>&1 &
    CLIENT1_PID=$!

    sleep 5

    echo 'Start profiling client 0'
    python client/block_inference_cv.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}" --device-id "${GPU_ID}" -i "${GPU_ID}" -dbn "${EXP_SAVE_DIR}/batch_size"

    echo 'Cleaning up...'
    kill -9 $CLIENT1_PID

    echo 'Finish!'

    sleep 10
done