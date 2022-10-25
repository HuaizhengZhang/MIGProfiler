cp docker/* ./
docker build -t mig-perf/profiler:1.0 -f "ubuntu.Dockerfile" .
docker pull nvcr.io/nvidia/k8s/dcgm-exporter:2.4.7-2.6.11-ubuntu20.04

