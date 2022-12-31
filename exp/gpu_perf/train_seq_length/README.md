# Train Workload with Different Sequence Length on MIG Instances

- Task: Sequence Classification
- Model: BERT-base-cased
- Workload: Train
- Batch size: 128
- Dynamic max sequence length: 32, 64, 128, 256
- Software: Native PyTorch
- Device: A100 80GB

## 1. Block Request Workload

1. Image Classification

JSON benchmarking results saved at `./train_seq_length`.

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
