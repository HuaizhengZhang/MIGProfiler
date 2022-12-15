#! /usr/bin/env bash
GPU_ID=0
MODEL_NAME='bert-base-cased'
NUM_TEST_BATCHES=1000
#MIG_PROFILES=('1g.10gb' '2g.20gb' '3g.40gb' '4g.40gb' '7g.80gb')
MIG_PROFILES=('4g.40gb')
SEQ_LENS=(32 64 128 256)
BATCH_SIZE=32

EXP_SAVE_DIR="${PWD}"
cd ../../mig_perf/inference
export PYTHONPATH="${PWD}"

echo 'Enable MIG'
sudo nvidia-smi -i "${GPU_ID}" -mig 1
# Try different MIG profiles
for MIG_PROFILE in "${MIG_PROFILES[@]}"; do
  echo '=========================================================='
  echo " * MIG PROFILE = ${MIG_PROFILE}"
  echo '=========================================================='
  sudo nvidia-smi mig -i 0 -cgi "${MIG_PROFILE}" -C
  sleep 5

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
    python client/block_inference_nlp.py -b "${BATCH_SIZE}" -m "${MODEL_NAME}" -n "${NUM_TEST_BATCHES}" --seq_len "${SEQ_LEN}" \
      -i "${GPU_ID}" -mi 0 -dbn "${EXP_SAVE_DIR}/seq_length/${MIG_PROFILE}" --report-suffix "seq${SEQ_LEN}"

    echo 'Finish!'
    sleep 10
  done

  echo 'Stop DCGM'
  docker stop dcgm_exporter

  sudo nvidia-smi mig -i "${GPU_ID}" -dci
  sudo nvidia-smi mig -i "${GPU_ID}" -dgi

  sleep 10
done
echo 'Disable MIG'
sudo nvidia-smi -i "${GPU_ID}" -mig 0