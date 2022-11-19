docker run --rm --gpus 'device=0:0' --net mig_perf \
--name profiler --cap-add SYS_ADMIN --shm-size="15g" \
-v /home/migtest/MIGProfiler/data/:/workspace/data/  \
mig-perf/profiler:1.0 "model_name=vision_transformer" "workload=cv_infer" "gpu_i_id=0"