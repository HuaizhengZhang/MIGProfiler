docker run -d --rm --gpus all --net mig_perf -p 9400:9400  \
--name dcgm_exporter --cap-add SYS_ADMIN   nvcr.io/nvidia/k8s/dcgm-exporter:2.4.7-2.6.11-ubuntu20.04 \
-c 100 -f /etc/dcgm-exporter/dcp-metrics-included.csv -d i

docker run --rm --gpus 'device=0:0' --net mig_perf \
--name profiler --cap-add SYS_ADMIN --shm-size="15g" \
-v /root/places365_standard/:/workspace/places365/  \
-v /root/MIGProfiler/data/:/workspace/data/ \
-v /root/MIGProfiler/logs/:/workspace/logs/  \
mig-perf/profiler:1.0 "model_name={$1}" "workload={$2}"