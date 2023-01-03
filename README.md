# MIG Profiler

![GitHub](https://img.shields.io/github/license/MLSysOps/MIGProfiler)

MIGProfiler is a toolkit for benchmark study on NVIDIA [MIG](https://www.nvidia.com/en-sg/technologies/multi-instance-gpu/) techniques. It provides profiling on multiple deep learning training and inference tasks on MIG GPUs. 

MIGProfiler is featured for:
- 🎨 Support a lot of deep learning tasks and open-sourced models on a various of benchmark type
- 📈 Present **comprehensive** benchmark results
- 🐣 **Easy to use** with a configuration file (WIP)

*The project is under rapid development! Please check our [benchmark results](#benchmark-result-📈) and join us!*

- [Benchmark Website](#benchmark-website-📈)
- [Install](#install-📦️)
- [Quick Start](#quick-start-🚚)
- [Cite Us](#cite-us-🌱)
- [Contributors](#contributors-👥)
- [Acknowledgement](#ackowledgement)
- [License](#license)

## Benchmark Website 📈
 Coming soon!

## Install 📦️

### Manual install

Requirements:
- PyTorch with CUDA
- OpenCV
- Sanic
- Transformers
- Tqdm
- Prometheus client

```shell
# create virtual environment
conda create -n mig-perf python=3.8
conda activate mig-perf

# install required packages
conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c conda-forge opencv
pip install transformers
pip install sanic tqdm prometheus_client
```

### PyPI install
WIP

### Use Docker
WIP

## Quick Start 🚚
You can easily to profile on MIG GPU. Below are some common deep learning tasks to play with.
### 1. MIG training benchmark

We first create a `1g.10gb` MIG device
```shell
# enable MIG
sudo nvidia-smi -i 0 -mig 1
# create MIG instance
sudo nvidia-smi mig -cgi 1g.10gb -C
```

Start DCGM metric exporter
```shell
docker run -d --rm --gpus all --net mig_perf -p 9400:9400  \
    -v "${PWD}/mig_perf/profiler/client/dcp-metrics-included.csv:/etc/dcgm-exporter/customized.csv" \
    --name dcgm_exporter --cap-add SYS_ADMIN   nvcr.io/nvidia/k8s/dcgm-exporter:2.4.7-2.6.11-ubuntu20.04 \
    -c 500 -f /etc/dcgm-exporter/customized.csv -d f
```

Start to profile
```shell
cd mig_perf/profiler
export PYTHONPATH=$PWD
python train/train_cv.py --bs=32 --model=resnet50 --num_batches=500 --mig-device-id=0
```

Remeber to disable MIG after finish benchmark
```shell
sudo nvidia-smi -i 0 -dci
sudo nvidia-smi -i 0 -dgi
sudo nvidia-smi -i 0 -mig 0
```

### 2. MIG inference benchmark

Start DCGM metric exporter
```shell
docker run -d --rm --gpus all --net mig_perf -p 9400:9400  \
    -v "${PWD}/mig_perf/profiler/client/dcp-metrics-included.csv:/etc/dcgm-exporter/customized.csv" \
    --name dcgm_exporter --cap-add SYS_ADMIN   nvcr.io/nvidia/k8s/dcgm-exporter:2.4.7-2.6.11-ubuntu20.04 \
    -c 500 -f /etc/dcgm-exporter/customized.csv -d f
```

Start to profile
```shell
cd mig_perf/profiler
export PYTHONPATH=$PWD
python client/block_infernece_cv.py --bs=32 --model=resnet50 --num_batches=500 --mig-device-id=0
```

See more benchmark experiments in [`./exp`](./exp).

## Cite Us 🌱

```bibtex
@article{zhang2022migperf,
  title={MIGPerf: A Comprehensive Benchmark for Deep Learning Training and Inference Workloads on Multi-Instance GPUs},
  author={Zhang, Huaizheng and Li, Yuanming and Xiao, Wencong and Huang, Yizheng and Di, Xing and Yin, Jianxiong and See, Simon and Luo, Yong and Lau, Chiew Tong and You, Yang},
  journal={arXiv preprint arXiv:2301.00407},
  year={2023}
}
```

## Contributors 👥

- Yuanming Li
- Huaizheng Zhang
- Yizheng Huang
- Xing Di

## Ackowledgement
Special thanks to Aliyun and NVIDIA AI Tech Center to provide MIG GPU server for benchmarking.

## License
This repository is open-sourced under [MIT License](./LICENSE).