#! /usr/bin/env bash
GPU_ID=1
MODEL_NAME=resnet50
TEST_TIME=30
ARRIVAL_RATES=(25 50 75 100 125 150 200)
batch_size=1

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

for ARRIVAL_RATE in "${ARRIVAL_RATES[@]}"; do
  echo 'Start server0'
  MAX_BATCH_SIZE="${batch_size}" MAX_WAIT_TIME=10 MODEL_NAME="${MODEL_NAME}" TASK="image_classification" DEVICE_ID="${GPU_ID}" \
    python server/app.py > /dev/null 2>&1 &
  SERVER0_PID=$!

  echo 'Start server1'
  MAX_BATCH_SIZE="${batch_size}" MAX_WAIT_TIME=10 MODEL_NAME="${MODEL_NAME}" TASK="image_classification" PORT=50076 DEVICE_ID="${GPU_ID}" \
    python server/app.py > /dev/null 2>&1 &
  SERVER1_PID=$!

  echo 'Start server2'
  MAX_BATCH_SIZE="${batch_size}" MAX_WAIT_TIME=10 MODEL_NAME="${MODEL_NAME}" TASK="image_classification" PORT=50077 DEVICE_ID="${GPU_ID}" \
    python server/app.py > /dev/null 2>&1 &
  SERVER2_PID=$!

  echo 'Start server3'
  MAX_BATCH_SIZE="${batch_size}" MAX_WAIT_TIME=10 MODEL_NAME="${MODEL_NAME}" TASK="image_classification" PORT=50078 DEVICE_ID="${GPU_ID}" \
    python server/app.py > /dev/null 2>&1 &
  SERVER3_PID=$!

  sleep 5

  echo 'Start profiling client 1'
  python client/pytorch_cv_client.py --url 'http://localhost:50076' -r "${ARRIVAL_RATE}" -dbn "${EXP_SAVE_DIR}" \
    --report-suffix 'client1' -b "${batch_size}" -t "${TEST_TIME}" -P -m "${MODEL_NAME}" -i "${GPU_ID}" > /dev/null 2>&1 &
  CLIENT1_PID=$!

  echo 'Start profiling client 2'
  python client/pytorch_cv_client.py --url 'http://localhost:50077' -r "${ARRIVAL_RATE}" -dbn "${EXP_SAVE_DIR}" \
    --report-suffix 'client2' -b "${batch_size}" -t "${TEST_TIME}" -P -m "${MODEL_NAME}" -i "${GPU_ID}" > /dev/null 2>&1 &
  CLIENT2_PID=$!

  echo 'Start profiling client 3'
  python client/pytorch_cv_client.py --url 'http://localhost:50078' -r "${ARRIVAL_RATE}" -dbn "${EXP_SAVE_DIR}" \
    --report-suffix 'client3' -b "${batch_size}" -t "${TEST_TIME}" -P -m "${MODEL_NAME}" -i "${GPU_ID}" > /dev/null 2>&1 &
  CLIENT3_PID=$!

  echo 'Start profiling client 0'
  python client/pytorch_cv_client.py -r "${ARRIVAL_RATE}" -dbn "${EXP_SAVE_DIR}" \
    --report-suffix 'client0' -b "${batch_size}" -t "${TEST_TIME}" -P -m "${MODEL_NAME}" -i "${GPU_ID}"

  echo 'Wait all client to finish'
  wait $CLIENT1_PID
  wait $CLIENT2_PID
  wait $CLIENT3_PID

  echo 'Cleaning up...'
  kill $SERVER0_PID $SERVER1_PID $SERVER2_PID $SERVER3_PID

  echo 'Finish!'

  sleep 10
done

echo 'Disable MPS'
echo quit | nvidia-cuda-mps-control

echo 'Shutdown DCGM'
docker stop dcgm_exporter

echo 'Set GPU compute mode to DEFAULT'
sudo nvidia-smi -i "${GPU_ID}" -c DEFAULT
