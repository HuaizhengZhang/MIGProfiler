model_names=('distil_v1' 'distil_v2' 'MiniLM' 'mpnet' 'bert-base')
group_id=2
profile_ids=(0 9 14 19)
dcgm_result_dir="save_dir=/root/A100-benchmark/data/infer/nlp/"
declare -A profile_device
profile_device[0]='MIG-29f07255-51c0-5691-82c4-cbc57760ff63'
profile_device[9]='MIG-ad654d5e-db91-50e1-bcf4-1e9c24f825ad'
profile_device[14]='MIG-ea08ed7b-a485-5967-9c78-2fa6e548c43a'
profile_device[19]='MIG-617584ed-80ed-5abf-a4bd-a5252a8433fa'

declare -A profile_name
profile_name[0]='7g.80gb'
profile_name[9]='3g.40gb'
profile_name[14]='2g.20gb'
profile_name[19]='1g.10gb'

batch_sizes=(1 2 4 8 16 32 64 128 256)
sequence_lengths=(32 128 256 512) # seq_length 64 is already tested in batch_size comparison testing

function batch_size_benchmark {
  for batch_size in ${batch_sizes[*]}
  do
      echo "model_name=$model_name, profile_device=${profile_name[$1]}, batch_size=$batch_size, seq_length=64"
      CUDA_VISIBLE_DEVICES=${profile_device[$1]} python A100-nlp-infer_test.py \
      "model_name=${model_name}" "mig_profile=${profile_name[$1]}" "batch_size=${batch_size}" "seq_length=64"
  done
  return 0
}

function seq_length_benchmark {
  for seq_length in ${sequence_lengths[*]}
  do
      echo "model_name=$model_name, profile_device=${profile_name[$1]}, batch_size=32, seq_length=$seq_length"
      CUDA_VISIBLE_DEVICES=${profile_device[$1]} python A100-nlp-infer_test.py \
      "model_name=${model_name}" "mig_profile=${profile_name[$1]}" "seq_length=${seq_length}" "batch_size=32"
  done
  return 0
}

for model_name in ${model_names[*]}
do
  for profile_id in ${profile_ids[*]}
  do
    sleep 10
    sudo nvidia-smi mig -dci
    sudo nvidia-smi mig -dgi
    sudo nvidia-smi mig -cgi "${profile_id}" -C
    python dcgm_recorder.py "group_id=${group_id}" $dcgm_result_dir&
    pid=$!
    echo "dcgm process(pid=$pid) is running"
    batch_size_benchmark "$profile_id"
    #seq_length_benchmark "$profile_id"
    kill $pid
  done
done