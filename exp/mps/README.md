# MPS Experiment

## Start MPS

```shell
#mkdir mps_tmp
#export CUDA_MPS_LOG_DIRECTORY=mps_tmp/log 
#export CUDA_MPS_PIPE_DIRECTORY=$PWD/mps_tmp 
sudo nvidia-cuda-mps-control -d
```
Check the service started
```shell
pidof nvidia-cuda-mps-control
```

## Stop MPS
```shell
echo quit | sudo nvidia-cuda-mps-control
```
