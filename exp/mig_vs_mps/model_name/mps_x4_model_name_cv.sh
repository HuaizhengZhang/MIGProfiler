#! /usr/bin/env bash
GPU_ID=1
# 'resnet152' OOM
MODEL_NAMES=('resnet18' 'resnet34' 'resnet50' 'resnet101')
NUM_TEST_BATCHES=1000
BATCH_SIZE=32

BASE_DIR=$(realpath $0 | xargs dirname)
EXP_SAVE_DIR="${BASE_DIR}/mps_x4"
PYTHON_EXECUTION_ROOT="${BASE_DIR}/../../../mig_perf/profiler"
DCGM_EXPORTER_METRICS_PATH="${PYTHON_EXECUTION_ROOT}/client/dcp-metrics-included.csv:/etc/dcgm-exporter/customized.csv"
cd "${PYTHON_EXECUTION_ROOT}"
export PYTHONPATH="${PYTHON_EXECUTION_ROOT}"

echo 'Set GPU compute mode to EXCLUSIVE_PROCESS'
sudo nvidia-smi -i "${GPU_ID}" -c EXCLUSIVE_PROCESS

echo 'Enable MPS'
nvidia-cuda-mps-control -d
echo "MPS control running at $(pidof nvidia-cuda-mps-control)"

echo 'Start DCGM'
docker run -d --rm --gpus all --net mig_perf -p 9400:9400  \
  -v "${DCGM_EXPORTER_METRICS_PATH}:/etc/dcgm-exporter/customized.csv" \
  --name dcgm_exporter --cap-add SYS_ADMIN   nvcr.io/nvidia/k8s/dcgm-exporter:2.4.7-2.6.11-ubuntu20.04 \
  -c 500 -f /etc/dcgm-exporter/customized.csv -d f
sleep 3
docker ps

# iterate through batch size list
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Model name ${MODEL_NAME}"
    echo 'Start client 1'
    python client/block_inference_cv.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}" \
      -i "${GPU_ID}" -dbn "${EXP_SAVE_DIR}" --report-suffix 'client1' > /dev/null 2>&1 &
    CLIENT1_PID=$!

    echo 'Start client 2'
    python client/block_inference_cv.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}" \
      -i "${GPU_ID}" -dbn "${EXP_SAVE_DIR}" --report-suffix 'client2' > /dev/null 2>&1 &
    CLIENT2_PID=$!

    echo 'Start client 3'
    python client/block_inference_cv.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}" \
      -i "${GPU_ID}" -dbn "${EXP_SAVE_DIR}" --report-suffix 'client3' > /dev/null 2>&1 &
    CLIENT3_PID=$!

    sleep 5

    echo 'Start profiling client 0'
    python client/block_inference_cv.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}" \
      -i "${GPU_ID}" -dbn "${EXP_SAVE_DIR}" --report-suffix 'client0'

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

echo 'Set GPU compute mode to DEFAULT'
sudo nvidia-smi -i "${GPU_ID}" -c DEFAULT
