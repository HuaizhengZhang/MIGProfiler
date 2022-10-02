sudo nvidia-smi mig -dci -i "$1"
sudo nvidia-smi mig -dgi- i "$1"
sudo nvidia-smi mig -cgi "$2" -i "$1" -C