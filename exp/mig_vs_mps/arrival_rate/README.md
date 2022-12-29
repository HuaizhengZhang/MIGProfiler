# MIG vs. MPS Multi-instacnes with Different Request Arrival Rate on NVIDIA A30 GPU

- Task: Image Classification
- Model: ResNet-50
- Workload: Async
- Arrival rate: 25, 50, 75, 100, 125, 150, 200
- Batch size: 1
- Software: PyTorch Infernece Server
  - Dynamic batching: OFF
  - Max request wait time: 10s
- Device: A30 24GB
- Model instances: 4

## 1. Image Classification

JSON benchmark results will be saved at `./1g.6gbx4` (for MIG) and `./mps_x4` (for MPS).

Benchmark on MIG-enabled four `1g.6gb` GPU instances (GI):
```shell
bash mig_1gx4_arrival_rate_cv.sh
```

Benchmark on MPS enabled GPU:
```shell
bash mps_x4_arrival_rate_cv.sh
```
