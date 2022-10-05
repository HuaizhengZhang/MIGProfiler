FROM nvcr.io/nvidia/cuda:11.6.0-base-ubuntu20.04

LABEL io.k8s.display-name="NVIDIA DCGM"
LABEL name="NVIDIA DCGM"
LABEL summary="Manage NVIDIA GPUs"

ARG DCGM_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y systemd && apt-get install -y --no-install-recommends \
    datacenter-gpu-manager libcap2-bin && apt-get purge --autoremove -y openssl

# Required for full gpu access
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,compat32
# disable all constraints on the configurations required by NVIDIA container toolkit
ENV NVIDIA_DISABLE_REQUIRE="true"
ENV NVIDIA_VISIBLE_DEVICES=all

ENV NO_SETCAP=
# open port 5555 to other containers ie, dcgm-exporter
EXPOSE 5555

COPY start_host_engine.sh start.sh
# ENTRYPOINT sh -c "sh start.sh" && dcgmi discovery -l
CMD sh -c "sh start.sh" && bash
