# $1: gpu_id, $2: mig_profile
sudo nvidia-smi mig -i "$1" -dci
sudo nvidia-smi mig -i "$1" -dgi
sudo nvidia-smi mig -i "$1" -cgi "$2" -C