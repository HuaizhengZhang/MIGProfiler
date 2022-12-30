# Infernece Workload with Different Batch Size on MIG Instances

- Task: Image Classification / Sequence Classification
- Model: ResNet-50 / BERT-base-cased
- Workload: Block / Async (4 threads)
- Batch size: 1, 2, 4, 8, 16, 32, 64
- Fixed sequence length: 64
- Software: Native PyTorch (Infernece Only)
- Device: A100 80GB

## 1. Block Request Workload

1. Image Classification

JSON benchmarking results saved at `./block_request`.

Benchmark with different MIG GPU Instance (GI):  
`1g.10gb`, `2g.20gb`, `3g.40gb`, `4g.40gb`, `7g.80gb`.
```shell
bash mig_diff_gi_batch_size_block_cv.sh
```

Benchmark without MIG enabled
```shell
bash no_mig_batch_size_block_cv.sh
```

2. Text Sequence Classification

Benchmark with different MIG GPU Instance (GI):  
`1g.10gb`, `2g.20gb`, `3g.40gb`, `4g.40gb`, `7g.80gb`:
```shell
bash mig_diff_gi_batch_size_block_nlp.sh
```

Benchmark without MIG enabled
```shell
bash no_mig_batch_size_block_nlp.sh
```

## 2. Async Request Workload

JSON benchmarking results saved at `./async_request`.

1. Image Classification

Benchmark with different MIG GPU Instances (GIs):  
`1g.10gb`, `2g.20gb`, `3g.40gb`, `4g.40gb`, `7g.80gb`.
```shell
bash mig_diff_gi_batch_size_async_nlp.sh
```

Benchmark without MIG enabled
```shell
bash no_mig_batch_size_async_nlp.sh
```

2. Text Sequence Classification

Benchmark with different MIG GPU Instance (GI):  
`1g.10gb`, `2g.20gb`, `3g.40gb`, `4g.40gb`, `7g.80gb`.
```shell
bash mig_diff_gi_batch_size_async_nlp.sh
```

Benchmark without MIG enabled
```shell
bash no_mig_batch_size_async_nlp.sh
```
