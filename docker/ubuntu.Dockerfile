FROM nvcr.io/nvidia/cuda:11.6.0-base-ubuntu20.04
FROM pytorch/pytorch
COPY ./mig_perf ./mig_perf
COPY ./requirements.txt ./requirements.txt
WORKDIR ./
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
ENV PYTHONPATH ./
ENV TORCH_HOME ./data/torch_models
ENV HF_HOME ./data/huggingface
ENTRYPOINT ["python", "mig_perf/profiler/profiler.py"]