FROM nvcr.io/nvidia/pytorch:22.10-py3
COPY ./mig_perf ./mig_perf
COPY ./requirements.txt ./requirements.txt
WORKDIR ./
RUN pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple
ENV PYTHONPATH ./
ENV TORCH_HOME ./data/torch_models
ENV HF_HOME ./data/huggingface
ENTRYPOINT ["python", "mig_perf/profiler/profiler.py"]