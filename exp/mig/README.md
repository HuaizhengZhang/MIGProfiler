# MIG Experiment

## 0. Setup

### 0.1. Setup GPU Compute Mode to DEFAULT (Only do once)

Reqire root:
```shell
nvidia-smi -i 0 -c DEFAULT
```
`-i` specified GPU ID.

Check compute mode:
```shell
nvidia-smi -i 0 -q | grep 'Compute Mode'
```

### 0.2. Enable MIG on the specific GPU
```shell
sudo nvidia-smi -i 0 -mig 1
```
Note that you need to stop all process using the GPU (e.g., dcgm).

### 0.3. Config MIG profile

```shell
sudo nvidia-smi mig -i 0 -cgi 2g.12gb,2g.12gb -C
```

Check the MIG GPU instances
```shell
sudo nvidia-smi mig -lgi
```

### 0.4. Stop MIG after experiment
```shell
sudo nvidia-smi mig -dci
sudo nvidia-smi mig -dgi
sudo nvidia-smi -mig 0
```

## 1. MIG GPU instances offline inference on A30 with different batch size

2 instances
```shell
bash mig_2+2_batch_size.sh
```

4 instances
```shell
bash mig_1+1+1+1_batch_size.sh
```

## 2. MIG GPU instances offline inference on A30 with different model name
4 instances
```shell
bash mig_1+1+1+1_model_name.sh
```
