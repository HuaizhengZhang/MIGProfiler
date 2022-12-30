# Infernece Workload with Different Text Sequnece Length on MIG Instances

- Task: Sequence Classification
- Model: BERT-base-cased
- Workload: Block / Async (4 threads)
- Batch size: 32
- Fixed sequence length: 32, 64, 128, 256
- Software: Native PyTorch (Infernece Only)
- Device: A100 80GB

## 1. Block Request Workload

JSON benchmarking results saved at `./block_request`.

Benchmark with different MIG GPU Instance (GI):  
`1g.10gb`, `2g.20gb`, `3g.40gb`, `4g.40gb`, `7g.80gb`.
```shell
bash mig_diff_gi_seq_len_block_nlp.sh
```

Benchmark without MIG enabled
```shell
bash no_mig_seq_len_block_nlp.sh
```

## 2. Async Request Workload

JSON benchmarking results saved at `./async_request`.

Benchmark with different MIG GPU Instance (GI):  
`1g.10gb`, `2g.20gb`, `3g.40gb`, `4g.40gb`, `7g.80gb`.
```shell
bash mig_diff_gi_seq_len_async_nlp.sh
```

Benchmark without MIG enabled
```shell
bash no_mig_seq_len_async_nlp.sh
```
