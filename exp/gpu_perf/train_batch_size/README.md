# Training Workload with Different Text Batch Size on MIG Instances

- Task: Image Classifiction / Sequence Classification
- Model: ResNet-50 / BERT-base-cased
- Workload: Train
- Batch size: 16, 32, 64, 128, 256, 512
- Dynamic max sequence length (for NLP tasks): 64
- Software: Native PyTorch
- Device: A100 80GB

JSON benchmarking results saved at `./train_batch_size`.

1. Image Classification

Benchmark with different MIG GPU Instance (GI):  
`1g.10gb`, `2g.20gb`, `3g.40gb`, `4g.40gb`, `7g.80gb`.
```shell
bash mig_diff_gi_train_batch_size_cv.sh
```

Benchmark without MIG enabled
```shell
bash no_mig_train_batch_size_cv.sh
```

2. Text Sequence Classification

Benchmark with different MIG GPU Instance (GI):  
`1g.10gb`, `2g.20gb`, `3g.40gb`, `4g.40gb`, `7g.80gb`:
```shell
bash mig_diff_gi_train_batch_size_nlp.sh
```

Benchmark without MIG enabled
```shell
bash no_mig_gi_train_batch_size_nlp.sh
```
