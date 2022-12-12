#! /usr/bin/env bash
GPU_ID=0
MODEL_NAME=resnet18
NUM_TEST_BATCHES=1000
BATCH_SIZES=(1 2 4 8 16 32 64)

EXP_SAVE_DIR="${PWD}"
cd ../../mig_perf/inference
export PYTHONPATH="${PWD}"

# iterate through batch size list
for batch_size in "${BATCH_SIZES[@]}"; do
    echo "Batch size ${batch_size}"
    python client/block_inference.py -b "${batch_size}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}" --device-id "${GPU_ID}" -dbn "${EXP_SAVE_DIR}/batch_size"

    sleep 10
done
