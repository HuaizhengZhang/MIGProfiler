#! /usr/bin/env bash
GPU_ID=1
MODEL_NAME=resnet50
TEST_TIME=30
ARRIVAL_RATES=(25 50 75 100 125 150 200)
batch_size=1

EXP_SAVE_DIR="${PWD}"
cd ../../mig_perf/inference
export PYTHONPATH="${PWD}"

for ARRIVAL_RATE in "${ARRIVAL_RATES[@]}"; do
  echo 'Start server0'
  MAX_BATCH_SIZE="${batch_size}" MAX_WAIT_TIME=10 MODEL_NAME="${MODEL_NAME}" TASK="image_classification" DEVICE_ID="${GPU_ID}" python server/app.py > /dev/null 2>&1 &
  SERVER0_PID=$!

  echo 'Start server1'
  MAX_BATCH_SIZE="${batch_size}" MAX_WAIT_TIME=10 MODEL_NAME="${MODEL_NAME}" TASK="image_classification" PORT=50076 DEVICE_ID="${GPU_ID}" python server/app.py > /dev/null 2>&1 &
  SERVER1_PID=$!

  echo 'Start server2'
  MAX_BATCH_SIZE="${batch_size}" MAX_WAIT_TIME=10 MODEL_NAME="${MODEL_NAME}" TASK="image_classification" PORT=50077 DEVICE_ID="${GPU_ID}" python server/app.py > /dev/null 2>&1 &
  SERVER2_PID=$!

  echo 'Start server3'
  MAX_BATCH_SIZE="${batch_size}" MAX_WAIT_TIME=10 MODEL_NAME="${MODEL_NAME}" TASK="image_classification" PORT=50078 DEVICE_ID="${GPU_ID}" python server/app.py > /dev/null 2>&1 &
  SERVER3_PID=$!

  sleep 5

  echo 'Start profiling client 1'
  python client/pytorch_cv_client.py --url 'http://localhost:50076' -r "${ARRIVAL_RATE}" -dbn "${EXP_SAVE_DIR}/arrival_rate_4_instance" --report-suffix 'client1' -b "${batch_size}" -t "${TEST_TIME}" -P -m "${MODEL_NAME}" -i "${GPU_ID}" > /dev/null 2>&1 &
  CLIENT1_PID=$!

  echo 'Start profiling client 2'
  python client/pytorch_cv_client.py --url 'http://localhost:50077' -r "${ARRIVAL_RATE}" -dbn "${EXP_SAVE_DIR}/arrival_rate_4_instance" --report-suffix 'client2' -b "${batch_size}" -t "${TEST_TIME}" -P -m "${MODEL_NAME}" -i "${GPU_ID}" > /dev/null 2>&1 &
  CLIENT2_PID=$!

  echo 'Start profiling client 3'
  python client/pytorch_cv_client.py --url 'http://localhost:50078' -r "${ARRIVAL_RATE}" -dbn "${EXP_SAVE_DIR}/arrival_rate_4_instance" --report-suffix 'client3' -b "${batch_size}" -t "${TEST_TIME}" -P -m "${MODEL_NAME}" -i "${GPU_ID}" > /dev/null 2>&1 &
  CLIENT3_PID=$!

  echo 'Start profiling client 0'
  python client/pytorch_cv_client.py -r "${ARRIVAL_RATE}" -dbn "${EXP_SAVE_DIR}/arrival_rate_4_instance" --report-suffix 'client0' -b "${batch_size}" -t "${TEST_TIME}" -P -m "${MODEL_NAME}" -i "${GPU_ID}"

  echo 'Wait all client to finish'
  wait $CLIENT1_PID
  wait $CLIENT2_PID
  wait $CLIENT3_PID

  echo 'Cleaning up...'
  kill $SERVER0_PID $SERVER1_PID $SERVER2_PID $SERVER3_PID

  echo 'Finish!'

  sleep 10
done
