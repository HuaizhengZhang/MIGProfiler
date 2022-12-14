#! /usr/bin/env bash
### Disable MPS, enable MIG
# echo quit | nvidia-cuda-mps-control
# sudo nvidia-smi -i 0 -mig 1
# sudo nvidia-smi mig -i 0 -cgi 1g.6gb,1g.6gb,1g.6gb,1g.6gb -C
GPU_ID=0
MODEL_NAMES=('resnet18' 'resnet34' 'resnet50' 'resnet101')
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
    echo 'Start dummy client 1'
    python client/block_inference_cv.py --dry-run  -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n 1048575  \
      -i "${GPU_ID}" -mi 1  > /dev/null 2>&1 &
    CLIENT1_PID=$!

    echo 'Start dummy client 2'
    python client/block_inference_cv.py --dry-run  -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n 1048575  \
      -i "${GPU_ID}" -mi 2  > /dev/null 2>&1 &
    CLIENT2_PID=$!

    echo 'Start dummy client 3'
    python client/block_inference_cv.py --dry-run  -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n 1048575  \
      -i "${GPU_ID}" -mi 3  > /dev/null 2>&1 &
    CLIENT3_PID=$!

    sleep 5

    echo 'Start profiling client 0'
    python client/block_inference_cv.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}" \
      -i "${GPU_ID}" -mi 0  -dbn "${EXP_SAVE_DIR}/model_name1x4"

    echo 'Cleaning up...'
    kill -9 $CLIENT1_PID $CLIENT2_PID $CLIENT3_PID

    echo 'Finish!'

    sleep 10
done

echo 'Stop DCGM'
docker stop dcgm_exporter

echo 'Disable MIG'
sudo nvidia-smi mig -i "${GPU_ID}" -dci
sudo nvidia-smi mig -i "${GPU_ID}" -dgi
sudo nvidia-smi -i "${GPU_ID}" -mig 0
