FROM nvcr.io/nvidia/cuda:11.6.0-base-ubuntu20.04
FROM pytorch/pytorch
COPY ./ ./
WORKDIR ./
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
ENV PYTHONPATH ./
ENTRYPOINT ["python", "mig_perf/profiler/profiler.py"]