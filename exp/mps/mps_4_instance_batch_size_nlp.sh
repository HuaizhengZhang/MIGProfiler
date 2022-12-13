#! /usr/bin/env bash
GPU_ID=1
MODEL_NAME=resnet50
NUM_TEST_BATCHES=1000
BATCH_SIZES=(1 2 4 8 16 32 64)
SEQ_LEN=64

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
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    echo "Batch size ${BATCH_SIZE}"
    echo 'Start client 1'
    python client/block_inference_nlp.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}" -seq_len "${SEQ_LEN}" \
      -i "${GPU_ID}" -dbn "${EXP_SAVE_DIR}/batch_size_4_instance" --report-suffix "client1"  > /dev/null 2>&1 &
    CLIENT1_PID=$!

    echo 'Start client 2'
    python client/block_inference_nlp.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}" -seq_len "${SEQ_LEN}" \
       -i "${GPU_ID}" -dbn "${EXP_SAVE_DIR}/batch_size_4_instance" --report-suffix "client2" > /dev/null 2>&1 &
    CLIENT2_PID=$!

    echo 'Start client 3'
    python client/block_inference_nlp.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}" -seq_len "${SEQ_LEN}" \
      -i "${GPU_ID}" -dbn "${EXP_SAVE_DIR}/batch_size_4_instance" --report-suffix "client3" > /dev/null 2>&1 &
    CLIENT3_PID=$!

    echo 'Start profiling client 0'
    python client/block_inference_nlp.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}" -seq_len "${SEQ_LEN}" \
      -i "${GPU_ID}" -dbn "${EXP_SAVE_DIR}/batch_size_4_instance" --report-suffix "client0"

    echo 'Wait clients to finish...'
    wait $CLIENT1_PID
    wait $CLIENT2_PID
    wait $CLIENT3_PID

    echo 'Finish!'

    sleep 10
done

echo 'Disable MPS'
echo quit | nvidia-cuda-mps-control

echo 'Shutdown DCGM'
docker stop dcgm_exporter
