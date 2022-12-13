#! /usr/bin/env bash
### Disable MPS, enable MIG
# echo quit | nvidia-cuda-mps-control
# sudo nvidia-smi -i 0 -mig 1
# sudo nvidia-smi mig -i 0 -cgi 1g.6gb,1g.6gb,1g.6gb,1g.6gb -C
GPU_ID=0
# Note: GPU Instance ID (GPU_I_ID by DCGM) is different from GPU instance device ID. 
# GPU instance ID starts from 1
GPU_INSTANCE0_UUID='MIG-8b9c1298-b163-5ea0-b796-9f12de8b362a'
GPU_INSTANCE1_UUID='MIG-6dd9381e-80bd-5581-9702-563ef12adf3a'
GPU_INSTANCE2_UUID='MIG-6d04a28e-8852-564c-9193-e71ed27d7640'
GPU_INSTANCE3_UUID='MIG-f725e1f8-4463-514b-86b1-86532a7fe70c'
MODEL_NAMES=('resnet18' 'resnet34' 'resnet50' 'resnet101')
NUM_TEST_BATCHES=1000
BATCH_SIZE=32

EXP_SAVE_DIR="${PWD}"
cd ../../mig_perf/inference
export PYTHONPATH="${PWD}"

# iterate through batch size list
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Batch size ${BATCH_SIZE}"
    echo 'Start dummy client 1'
    python client/block_inference_cv.py --dry-run  -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n 1048575  \
      --device-id "${GPU_INSTANCE1_UUID}" -i "${GPU_ID}" -gi 4  > /dev/null 2>&1 &
    CLIENT1_PID=$!

    echo 'Start dummy client 2'
    python client/block_inference_cv.py --dry-run  -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n 1048575  \
      --device-id "${GPU_INSTANCE2_UUID}" -i "${GPU_ID}" -gi 5  > /dev/null 2>&1 &
    CLIENT2_PID=$!

    echo 'Start dummy client 3'
    python client/block_inference_cv.py --dry-run  -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n 1048575  \
      --device-id "${GPU_INSTANCE3_UUID}" -i "${GPU_ID}" -gi 6  > /dev/null 2>&1 &
    CLIENT3_PID=$!

    sleep 5

    echo 'Start profiling client 0'
    python client/block_inference_cv.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}" \
      --device-id "${GPU_INSTANCE0_UUID}" -i "${GPU_ID}" -gi 3  -dbn "${EXP_SAVE_DIR}/model_name1x4"

    echo 'Cleaning up...'
    kill -9 $CLIENT1_PID $CLIENT2_PID $CLIENT3_PID

    echo 'Finish!'

    sleep 10
done
