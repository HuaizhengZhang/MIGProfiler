model_names=('distil_v1' 'distil_v2' 'MiniLM' 'mpnet' 'bert-base')
profile_ids=(0 9 14 19)
declare -A map
map[0]=0
map[9]=2
map[14]=5
map[19]=13
for model_name in ${model_names[*]}
do
  for profile_id in ${profile_ids[*]}
  do
    sudo nvidia-smi mig -cgi "${profile_id}" -C
    CUDA_VISIBLE_DEVICES=MIG-GPU-6a15f30b-4e5e-ba37-fc16-4b2f0c9111f1/${map[$profile_id]}/0 python A100-infer_test.py "model_name=${model_name}" "fixed_time=True" "mig_profile_id=${profile_id}"\
                              "result_dir=/root/infer_test/A100-infer_results/data/infer/${model_name}/"\
                              "test_args=[{seq_length: 64, batch_size: 1},{seq_length: 64, batch_size: 2},{seq_length: 64, batch_size: 4},{seq_length: 64,batch_size: 8},{seq_length: 64, batch_size: 16},{seq_length: 64, batch_size: 32},{seq_length: 64, batch_size: 64},{seq_length: 64, batch_size: 128},{seq_length: 64, batch_size: 256}]"\
                              "hydra.run.dir=/root/infer_test/logs/infer/${model_name}/bsz_profile${profile_id}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}"
    sudo nvidia-smi mig -dci
    sudo nvidia-smi mig -dgi
  done
done