# MIG vs. MPS Multi-instacnes with Different Model Name on NVIDIA A30 GPU

- Task: Image Classification / Sequence Classification
- Model Famliy:  
  &nbsp;&nbsp; ResNet-18, ResNet-34, ResNet-50, ResNet-101 /  
  &nbsp;&nbsp; Distilbert-base-cased, BERT-base-cased, BERT-large-cased
- Workload: Block
- Batch size: 1, 2, 4, 8, 16, 32, 64
- Fixed sequence length: 64
- Software: Native PyTorch (Infernece Only)
- Device: A30 24GB
- Model instances: 4

## 1. Image Classification

JSON benchmark results will be saved at `./1g.6gbx4` (for MIG) and `./mps_x4` (for MPS).

Benchmark on MIG-enabled four `1g.6gb` GPU instances (GI):
```shell
bash mig_1gx4_model_name_cv.sh
```

Benchmark on MPS enabled GPU:
```shell
bash mps_x4_batch_size_cv.sh
```

## 2. Text Sequence Classification

Benchmark on MIG-enabled four `1g.6gb` GPU instances (GI):
```shell
bash mig_1gx4_model_name_nlp.sh`
```

Benchmark on MPS-enabled GPU:
```shell
bash mps_x4_model_name_nlp.sh
```
