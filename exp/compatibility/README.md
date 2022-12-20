# MIG Software Compatibility

## Install

### Jax
- CuDNN >= 8.2
- CUDA >= 11.4

First check your CUDA and CuDNN version:
```shell
# CUDA version
nvcc --version
# CuDNN version
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A2
```

```shell
# Installs the wheel compatible with Cuda >= 11.8 and cudnn >= 8.6
pip install "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Installs the wheel compatible with Cuda >= 11.4 and cudnn >= 8.2
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
