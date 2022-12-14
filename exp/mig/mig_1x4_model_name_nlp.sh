#! /usr/bin/env bash
GPU_ID=0
MODEL_NAMES=('distilbert-base-cased' 'bert-base-cased' 'bert-large-cased')
NUM_TEST_BATCHES=1000
BATCH_SIZE=32

EXP_SAVE_DIR="${PWD}"
cd ../../mig_perf/inference
export PYTHONPATH="${PWD}"

echo 'Enable MIG'
sudo nvidia-smi -i "${GPU_ID}" -mig 1
sudo nvidia-smi mig -cgi 1g.6gb,1g.6gb,1g.6gb,1g.6gb -C

echo 'Start DCGM'
docker run -d --rm --gpus all --net mig_perf -p 9400:9400  \
  -v "${EXP_SAVE_DIR}/../../mig_perf/inference/client/dcp-metrics-included.csv:/etc/dcgm-exporter/customized.csv" \
  --name dcgm_exporter --cap-add SYS_ADMIN   nvcr.io/nvidia/k8s/dcgm-exporter:2.4.7-2.6.11-ubuntu20.04 \
  -c 500 -f /etc/dcgm-exporter/customized.csv -d f
sleep 3
docker ps

# iterate through batch size list
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Batch size ${BATCH_SIZE}"
    echo 'Start client 1'
    python client/block_inference_nlp.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}"  \
      -i "${GPU_ID}" -mi 1 -dbn "${EXP_SAVE_DIR}/model_name1x4" --report-suffix 'client1' > /dev/null 2>&1 &
    CLIENT1_PID=$!

    echo 'Start client 2'
    python client/block_inference_nlp.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}"  \
      -i "${GPU_ID}" -mi 2 -dbn "${EXP_SAVE_DIR}/model_name1x4" --report-suffix 'client2' > /dev/null 2>&1 &
    CLIENT2_PID=$!

    echo 'Start client 3'
    python client/block_inference_nlp.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}"  \
      -i "${GPU_ID}" -mi 3 -dbn "${EXP_SAVE_DIR}/model_name1x4" --report-suffix 'client3' > /dev/null 2>&1 &
    CLIENT3_PID=$!

    echo 'Start profiling client 0'
    python client/block_inference_nlp.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}" \
      -i "${GPU_ID}" -mi 0  -dbn "${EXP_SAVE_DIR}/model_name1x4" --report-suffix 'client0'

    echo 'Wait clients to finish...'
    wait $CLIENT1_PID
    wait $CLIENT2_PID
    wait $CLIENT3_PID

    echo 'Finish!'

    sleep 10
done

echo 'Stop DCGM'
docker stop dcgm_exporter

echo 'Disable MIG'
sudo nvidia-smi mig -i "${GPU_ID}" -dci
sudo nvidia-smi mig -i "${GPU_ID}" -dgi
sudo nvidia-smi -i "${GPU_ID}" -mig 0