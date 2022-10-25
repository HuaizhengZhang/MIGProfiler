from mig_perf.controller.controller import profile_plan
import subprocess


@profile_plan(gpu_id=0, mig_profiles=['1g.10gb', '2g.20gb', '3g.40gb', '4g.40gb', '7g.80gb'])
def single_instance_benchmark(
        model_name: str,
        workload: str,
        device_str: str,
        cv_task_dataset_path: str,
        result_save_path: str,
        logs_save_path: str,
        hugginface_home: str,
        torch_home: str,
):
    cmd = f"docker run --rm --gpus '{str(device_str)}' --net mig_perf \
    --name profiler --cap-add SYS_ADMIN --shm-size=\"15g\" \
    -v {str(cv_task_dataset_path)}:/workspace/places365/  \
    -v {str(result_save_path)}:/workspace/data/ \
    -v {str(logs_save_path)}:/workspace/logs/  \
    -v {str(hugginface_home)}:/workspace/huggingface/ \
    -v {str(torch_home)}:/workspace/torch_models/ \
    mig-perf/profiler:1.0 \"model_name={str(model_name)}\" \"workload={str(workload)}\""
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(p.communicate()[0].decode("utf-8"))
    p.terminate()


if __name__ == '__main__':
    single_instance_benchmark(
        model_name="vision_transformer",
        workload="cv_infer",
        device_str="device=0:0",
        cv_task_dataset_path="/root/places365_standard/",
        result_save_path="/root/MIGProfiler/data/",
        logs_save_path="/root/MIGProfiler/logs/"
    )