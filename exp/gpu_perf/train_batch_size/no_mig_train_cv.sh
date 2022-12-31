#! /usr/bin/env bash
GPU_ID=0
MODEL_NAME='resnet50'
NUM_TRAIN_BATCHES=500
MIG_PROFILES=('1g.10gb' '2g.20gb' '3g.40gb' '4g.40gb' '7g.80gb')
BATCH_SIZES=(16 32 64 128 256 512)
MIG_PROFILE='no_mig'

BASE_DIR=$(realpath $0 | xargs dirname)
EXP_SAVE_DIR="${BASE_DIR}/train_batch_size"
PYTHON_EXECUTION_ROOT="${BASE_DIR}/../../../mig_perf/inference"
DCGM_EXPORTER_METRICS_PATH="${PYTHON_EXECUTION_ROOT}/client/dcp-metrics-included.csv:/etc/dcgm-exporter/customized.csv"
cd "${PYTHON_EXECUTION_ROOT}"
export PYTHONPATH="${PYTHON_EXECUTION_ROOT}"

echo '=========================================================='
echo " * MIG PROFILE = ${MIG_PROFILE}" 
echo '=========================================================='

# Try different MIG profiles
echo 'Start DCGM'
docker run -d --rm --gpus all --net mig_perf -p 9400:9400  \
  -v "${DCGM_EXPORTER_METRICS_PATH}:/etc/dcgm-exporter/customized.csv" \
  --name dcgm_exporter --cap-add SYS_ADMIN nvcr.io/nvidia/k8s/dcgm-exporter:2.4.7-2.6.11-ubuntu20.04 \
  -c 500 -f /etc/dcgm-exporter/customized.csv -d f
sleep 3
docker ps

# iterate through batch size list
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
  echo "Batch size ${BATCH_SIZE}"
  echo 'Start profiling client 0'
  python train/train_cv.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TRAIN_BATCHES}" \
    -i "${GPU_ID}" -dbn "${EXP_SAVE_DIR}/${MIG_PROFILE}"

  echo 'Finish!'
  sleep 10
done

echo 'Stop DCGM'
docker stop dcgm_exporter
