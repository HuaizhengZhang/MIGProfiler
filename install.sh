echo "Download the Places365 standard easyformat"
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
echo "Start to extract files"
mkdir ./data/ && tar -xvf places365standard_easyformat.tar -C ./data/
echo "Finish extracting files"
cp docker/.dockerignore ./.dockerignore
cp docker/ubuntu.Dockerfile ./ubuntu.Dockerfile
echo "build mig-perf image"
docker build -t mig-perf/profiler:1.0 -f "ubuntu.Dockerfile" .
echo "pull dcgm-exporter image"
docker pull nvcr.io/nvidia/k8s/dcgm-exporter:2.4.7-2.6.11-ubuntu20.04
echo "MIGProfiler successfully downloaded"

