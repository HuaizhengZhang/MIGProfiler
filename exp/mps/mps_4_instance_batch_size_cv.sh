#! /usr/bin/env bash
GPU_ID=1
MODEL_NAME=resnet50
NUM_TEST_BATCHES=1000
BATCH_SIZES=(1 2 4 8 16 32 64)

EXP_SAVE_DIR="${PWD}"
cd ../../mig_perf/inference
export PYTHONPATH="${PWD}"

# iterate through batch size list
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    echo "Batch size ${BATCH_SIZE}"
    echo 'Start client 1'
    python client/block_inference_cv.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}" --device-id "${GPU_ID}" -i "${GPU_ID}" -dbn "${EXP_SAVE_DIR}/batch_size_4_instance" --report-suffix "client1"  > /dev/null 2>&1 &
    CLIENT1_PID=$!

    echo 'Start client 2'
    python client/block_inference_cv.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}"  --device-id "${GPU_ID}" -i "${GPU_ID}" -dbn "${EXP_SAVE_DIR}/batch_size_4_instance" --report-suffix "client2" > /dev/null 2>&1 &
    CLIENT2_PID=$!

    echo 'Start client 3'
    python client/block_inference_cv.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}"  --device-id "${GPU_ID}" -i "${GPU_ID}" -dbn "${EXP_SAVE_DIR}/batch_size_4_instance" --report-suffix "client3" > /dev/null 2>&1 &
    CLIENT3_PID=$!

    echo 'Start profiling client 0'
    python client/block_inference_cv.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}" --device-id "${GPU_ID}" -i "${GPU_ID}" -dbn "${EXP_SAVE_DIR}/batch_size_4_instance" --report-suffix "client0"

    echo 'Wait clients to finish...'
    wait $CLIENT1_PID
    wait $CLIENT2_PID
    wait $CLIENT3_PID

    echo 'Finish!'

    sleep 10
done
