model_names=('resnet50')
profile_ids=(0 9 14 19)
declare -A map
map[0]='MIG-29f07255-51c0-5691-82c4-cbc57760ff63'
map[9]='MIG-ad654d5e-db91-50e1-bcf4-1e9c24f825ad'
map[14]='MIG-ea08ed7b-a485-5967-9c78-2fa6e548c43a'
map[19]='MIG-617584ed-80ed-5abf-a4bd-a5252a8433fa'
sudo nvidia-smi mig -dci
sudo nvidia-smi mig -dgi
for model_name in ${model_names[*]}
do
  for profile_id in ${profile_ids[*]}
  do
    sudo nvidia-smi mig -cgi "${profile_id}" -C
    CUDA_VISIBLE_DEVICES=${map[$profile_id]} python A100-pretrain_test.py \
    "arch=${model_name}" "fixed_time=True" "mig_profile_id=${profile_id}"\
    "result_dir=/root/A100-benchmark/data/train/${model_name}/"\
    "batch_sizes=[8,16,32,64,128,256]"\
    "hydra.run.dir=/root/A100-benchmark/logs/train/${model_name}/bsz_profile${profile_id}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"
    sudo nvidia-smi mig -dci
    sudo nvidia-smi mig -dgi
  done
done