FROM nvcr.io/nvidia/cuda:11.6.0-base-ubuntu20.04
COPY ../ ./workspace
VOLUME ["data"]
WORKDIR ./workspace
RUN pip install -r ../requirements.txt

CMD python mig_perf/profiler/profiler.py

