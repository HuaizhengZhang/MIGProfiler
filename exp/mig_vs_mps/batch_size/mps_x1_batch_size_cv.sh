#! /usr/bin/env bash
GPU_ID=1
MODEL_NAME=resnet18
NUM_TEST_BATCHES=1000
BATCH_SIZES=(1 2 4 8 16 32 64)

BASE_DIR=$(realpath $0 | xargs dirname)
EXP_SAVE_DIR="${BASE_DIR}/mps_x1"
PYTHON_EXECUTION_ROOT="${BASE_DIR}/../../../mig_perf/inference"
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
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    echo 'Start profiling client 0'
    python client/block_inference_cv.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}" -i "${GPU_ID}" \
      -dbn "${EXP_SAVE_DIR}"

    echo 'Finish!'

    sleep 10
done

echo 'Disable MPS'
echo quit | nvidia-cuda-mps-control

echo 'Shutdown DCGM'
docker stop dcgm_exporter

echo 'Set GPU compute mode to DEFAULT'
sudo nvidia-smi -i "${GPU_ID}" -c DEFAULT
