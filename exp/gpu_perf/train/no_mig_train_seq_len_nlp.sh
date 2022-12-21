#! /usr/bin/env bash
GPU_ID=0
MODEL_NAME='bert-base-cased'
NUM_TRAIN_BATCHES=500
BATCH_SIZE=128
SEQ_LENS=(32 64 128 256)

EXP_SAVE_DIR="${PWD}"
cd ../../mig_perf/inference
export PYTHONPATH="${PWD}"

echo 'Start DCGM'
docker run -d --rm --gpus all --net mig_perf -p 9400:9400  \
  -v "${EXP_SAVE_DIR}/../../mig_perf/inference/client/dcp-metrics-included.csv:/etc/dcgm-exporter/customized.csv" \
  --name dcgm_exporter --cap-add SYS_ADMIN   nvcr.io/nvidia/k8s/dcgm-exporter:2.4.7-2.6.11-ubuntu20.04 \
  -c 500 -f /etc/dcgm-exporter/customized.csv -d f
sleep 3
docker ps

# iterate through batch size list
for SEQ_LEN in "${SEQ_LENS[@]}"; do
  echo "Sequence length ${SEQ_LEN}"
  echo 'Start profiling client 0'
  python train/train_nlp.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TRAIN_BATCHES}" --seq_len "${SEQ_LEN}" \
    -i "${GPU_ID}" -dbn "${EXP_SAVE_DIR}/train/seq_length/no_mig"

  echo 'Finish!'
  sleep 10
done

echo 'Stop DCGM'
docker stop dcgm_exporter