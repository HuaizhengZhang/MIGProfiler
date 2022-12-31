# MIG vs. MPS Multi-instacnes with Different Batch Size on NVIDIA A30 GPU

- Task: Image Classification / Sequence Classification
- Model: ResNet-50 / BERT-base-cased
- Workload: Block
- Batch size: 1, 2, 4, 8, 16, 32, 64
- Fixed sequence length: 64
- Software: Native PyTorch (Infernece Only)
- Device: A30 24GB

## 1. 1 model instance

JSON benchmark results will be saved at `./4g.24gbx1` (for MIG) and `./mps_x1` (for MPS).

<ol>
<li>Image Classification

Benchmark on MIG-enabled one `4g.24gb` GPU instance (GI):
```shell
bash mig_4gx1_batch_size_cv.sh
```

Benchmark on MPS enabled GPU:
```shell
bash mps_x1_batch_size_cv.sh
```

</li>
</ol>

## 2. 2 model instances

JSON benchmark results will be saved at `./2g.12gbx2` (for MIG) and `./mps_x2` (for MPS).

<ol>
<li> Image Classification

Benchmark on MIG-enabled two `2g.12gb` GPU instances (GI):
```shell
bash mig_2gx2_batch_size_cv.sh
```

Benchmark on MPS enabled GPU:
```shell
bash mps_x2_batch_size_cv.sh
```

</li>
</ol>

## 3. 4 model instances

JSON benchmark results will be saved at `./1g.6gbx4` (for MIG) and `./mps_x4` (for MPS).

<ol>
<li> Image Classification

Benchmark on MIG-enabled four `1g.6gb` GPU instances (GI):
```shell
bash mig_1gx4_batch_size_cv.sh`
```

Benchmark on MPS-enabled GPU:
```shell
bash mps_x4_batch_size_cv.sh
```
</li>

<li> Text Sequence Classification

Benchmark on MIG-enabled four `1g.6gb` GPU instances (GI):
```shell
bash mig_1gx4_batch_size_nlp.sh`
```

Benchmark on MPS-enabled GPU:
```shell
bash mps_x4_batch_size_nlp.sh
```
</li>
</ol>
