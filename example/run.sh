docker run -d --rm --gpus all --net mig_perf -p 9400:9400 \
--name dcgm_exporter --cap-add SYS_ADMIN   nvcr.io/nvidia/k8s/dcgm-exporter:2.4.7-2.6.11-ubuntu20.04 \
-c 100 -f /etc/dcgm-exporter/dcp-metrics-included.csv -d i
gpu_id=0
mig_profile=('1g.10gb' '2g.20gb' '3g.40gb' '4g.40gb' '7g.80gb')
for mig_profile in ${mig_profile[*]}
do
  sudo nvidia-smi mig -dci -i "$gpu_id"
  sudo nvidia-smi mig -dgi -i "$gpu_id"
  sudo nvidia-smi mig -cgi "$mig_profile" -i "$gpu_id" -C
  sleep 2
  docker run --rm --gpus 'device=0:0' --net mig_perf \
  --name profiler --cap-add SYS_ADMIN --shm-size="15g" \
  -v /root/places365_standard/:/workspace/places365/  \
  -v /root/MIGProfiler/data/:/workspace/data/ \
  -v /root/MIGProfiler/logs/:/workspace/logs/  \
  -v /root/.cache/torch/:/workspace/torch_models/ \
  -v /root/.cache/huggingface/:/workspace/huggingface/ \
  mig-perf/profiler:1.0 "model_name=vision_transformer" "workload=cv_infer"
done