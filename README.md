# MIG Profiler

![GitHub](https://img.shields.io/github/license/MLSysOps/MIGProfiler)

MIGProfiler is a toolkit for benchmark study on NVIDIA [MIG](https://www.nvidia.com/en-sg/technologies/multi-instance-gpu/) techniques. It provides profiling on multiple deep learning training and inference tasks on MIG GPUs. 

MIGProfiler is featured for:
- 🎨 Support a lot of deep learning tasks and open-sourced models on a various of benchmark type
- 📈 Present **comprehensive** benchmark results
- 🐣 **Easy to use** with a configuration file (WIP)

*The project is under rapid development! Please check our [benchmark website](#benchmark-website-) and join us!*

- [Benchmark Website](#benchmark-website-)
- [Install](##install-)
- [Quick Start](#quick-start-)
- [Cite Us](#cite-us-)
- [Contributors](#contributors-)
- [Acknowledgement](#ackowledgement)
- [License](#license)

## Benchmark Website 📈
 Coming soon!

## Install 📦️

### Install by PyPI
```
pip install migperf
```
⚠️ For Deep Learning (DL) framework ([PyTorch](https://pytorch.org/)) and its task-specific DL libraries like [Hugging Face Transformers](https://pypi.org/project/transformers/) and OpenCV, you may need self-installation, since these libraries have various dependencies for different users.

### Use Docker
WIP

### Manual build

Clone the repo:
```
git clone https://github.com/MLSysOps/MIGProfiler.git
```

It is recommended to create a virtual environment for testing:
```shell
conda create -n mig-perf python=3.8
conda activate mig-perf
```
Manually install the required packages (you should install the correct version):
```shell
conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c conda-forge opencv
pip install transformers
```

Finally, build `migperf` package:
```shell
pip install .
```

## Quick Start 🚚
You can easily to profile on MIG GPU. Below are some common deep learning tasks to play with.
### 1. MIG training benchmark

We first create a `1g.10gb` MIG device
```python
from migperf.controller import MIGController
# enable MIG
mig_controller = MIGController()
mig_controller.enable_mig(gpu_id=0)
# Create GPU instance
gi_status = mig_controller.create_gpu_instance('1g.10gb', create_ci=True)
print(gi_status)
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
python train/train_cv.py --bs=32 --model=resnet50 --mig-device-id=0 --max_train_steps=10 
```

Clean up after benchmarking
```python
from migperf.controller import MIGController
# disable MIG
mig_controller = MIGController()
mig_controller.destroy_compute_instance(gpu_id=0)
mig_controller.destroy_gpu_instance(gpu_id=0)
mig_controller.disable_mig(gpu_id=0)
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
python client/block_inference_cv.py --bs=32 --model=resnet50 --num_batches=500 --mig-device-id=0
```

See more benchmark experiments in [`./exp`](./exp).

### 3. Visualize

- [x] in notebook
- [ ] in Prometheus (under improvement)

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
