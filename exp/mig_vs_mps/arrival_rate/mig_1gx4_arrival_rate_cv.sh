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
MODEL_NAME=resnet50
TEST_TIME=30
ARRIVAL_RATES=(25 50 75 100 125 150 200)
batch_size=1

BASE_DIR=$(realpath $0 | xargs dirname)
EXP_SAVE_DIR="${BASE_DIR}/1g.6gbx4"
PYTHON_EXECUTION_ROOT="${BASE_DIR}/../../../mig_perf/profiler"
DCGM_EXPORTER_METRICS_PATH="${PYTHON_EXECUTION_ROOT}/client/dcp-metrics-included.csv:/etc/dcgm-exporter/customized.csv"
cd "${PYTHON_EXECUTION_ROOT}"
export PYTHONPATH="${PYTHON_EXECUTION_ROOT}"

echo 'Enable MIG'
sudo nvidia-smi -i "${GPU_ID}" -mig 1
echo 'Create MIG instances'
sudo nvidia-smi mig -cgi 1g.6gb,1g.6gb,1g.6gb,1g.6gb -C

echo 'Start DCGM'
docker run -d --rm --gpus all --net mig_perf -p 9400:9400  \
  -v "${DCGM_EXPORTER_METRICS_PATH}:/etc/dcgm-exporter/customized.csv" \
  --name dcgm_exporter --cap-add SYS_ADMIN nvcr.io/nvidia/k8s/dcgm-exporter:2.4.7-2.6.11-ubuntu20.04 \
  -c 500 -f /etc/dcgm-exporter/customized.csv -d f
sleep 3
docker ps

# iterate through batch size list
for ARRIVAL_RATE in "${ARRIVAL_RATES[@]}"; do
    echo 'Start server0'
  MAX_BATCH_SIZE="${batch_size}" MAX_WAIT_TIME=10 MODEL_NAME="${MODEL_NAME}" TASK="image_classification" DEVICE_ID="${GPU_INSTANCE0_UUID}" python server/app.py > /dev/null 2>&1 &
  SERVER0_PID=$!

  echo 'Start server1'
  MAX_BATCH_SIZE="${batch_size}" MAX_WAIT_TIME=10 MODEL_NAME="${MODEL_NAME}" TASK="image_classification" PORT=50076 DEVICE_ID="${GPU_INSTANCE1_UUID}" python server/app.py > /dev/null 2>&1 &
  SERVER1_PID=$!

  echo 'Start server2'
  MAX_BATCH_SIZE="${batch_size}" MAX_WAIT_TIME=10 MODEL_NAME="${MODEL_NAME}" TASK="image_classification" PORT=50077 DEVICE_ID="${GPU_INSTANCE2_UUID}" python server/app.py > /dev/null 2>&1 &
  SERVER2_PID=$!

  echo 'Start server3'
  MAX_BATCH_SIZE="${batch_size}" MAX_WAIT_TIME=10 MODEL_NAME="${MODEL_NAME}" TASK="image_classification" PORT=50078 DEVICE_ID="${GPU_INSTANCE3_UUID}" python server/app.py > /dev/null 2>&1 &
  SERVER3_PID=$!

  sleep 5

  echo 'Start profiling client 1'
  python client/pytorch_cv_client.py --url 'http://localhost:50076' -r "${ARRIVAL_RATE}" -dbn "${EXP_SAVE_DIR}" --report-suffix 'client1' -b "${batch_size}" -t "${TEST_TIME}" -P -m "${MODEL_NAME}" -i "${GPU_ID}" -gi 4 > /dev/null 2>&1 &
  CLIENT1_PID=$!

  echo 'Start profiling client 2'
  python client/pytorch_cv_client.py --url 'http://localhost:50077' -r "${ARRIVAL_RATE}" -dbn "${EXP_SAVE_DIR}" --report-suffix 'client2' -b "${batch_size}" -t "${TEST_TIME}" -P -m "${MODEL_NAME}" -i "${GPU_ID}" -gi 5 > /dev/null 2>&1 &
  CLIENT2_PID=$!

  echo 'Start profiling client 3'
  python client/pytorch_cv_client.py --url 'http://localhost:50078' -r "${ARRIVAL_RATE}" -dbn "${EXP_SAVE_DIR}" --report-suffix 'client3' -b "${batch_size}" -t "${TEST_TIME}" -P -m "${MODEL_NAME}" -i "${GPU_ID}" -gi 6 > /dev/null 2>&1 &
  CLIENT3_PID=$!

  echo 'Start profiling client 0'
  python client/pytorch_cv_client.py -r "${ARRIVAL_RATE}" -dbn "${EXP_SAVE_DIR}" --report-suffix 'client0' -b "${batch_size}" -t "${TEST_TIME}" -P -m "${MODEL_NAME}" -i "${GPU_ID}" -gi 3

  echo 'Wait all client to finish'
  wait $CLIENT1_PID
  wait $CLIENT2_PID
  wait $CLIENT3_PID

  echo 'Cleaning up...'
  kill $SERVER0_PID $SERVER1_PID $SERVER2_PID $SERVER3_PID

  echo 'Finish!'

  sleep 10
done

echo 'Stop DCGM'
docker stop dcgm_exporter

echo 'Destroy MIG instances'
sudo nvidia-smi mig -i "${GPU_ID}" -dci
sudo nvidia-smi mig -i "${GPU_ID}" -dgi
echo 'Disable MIG'
sudo nvidia-smi -i "${GPU_ID}" -mig 0
echo 'Reset GPU'
sudo nvidia-smi -i "${GPU_ID}" -r
