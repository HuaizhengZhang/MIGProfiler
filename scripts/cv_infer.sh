export PYTHONPATH="${PWD}"
model_names=('vision_transformer' 'resnet50') # the models to be tested
group_id=2 # the dcgm group if you are going to monitor
profile_ids=(9 14 19) # the profile id of the mig partition
dcgm_result_dir="save_dir=/root/A100-benchmark/data/infer/cv/" # the gpu metric result are saved here
declare -A profile_device
# mig device name is different from gpu to gpu, assgin the correct instances names for your own experiments.
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

function batch_size_benchmark {
  for batch_size in ${batch_sizes[*]}
  do
      echo "model_name=$model_name, profile_device=${profile_name[$1]}, batch_size=$batch_size"
      CUDA_VISIBLE_DEVICES=${profile_device[$1]} python ./src/A100-cv-infer_test.py \
      "model_name=${model_name}" "mig_profile=${profile_name[$1]}" "batch_size=${batch_size}"
  done
  return 0
}
# run experiments on all assigned mig profiles and models
for model_name in ${model_names[*]}
do
  for profile_id in ${profile_ids[*]}
  do
    sleep 10
    sudo nvidia-smi mig -dci
    sudo nvidia-smi mig -dgi
    sudo nvidia-smi mig -cgi "${profile_id}" -C
    python ./src/dcgm_recorder.py "group_id=${group_id}" $dcgm_result_dir || exit &  # run dcgm monitor
    pid=$!
    echo "dcgm process(pid=$pid) is running"
    batch_size_benchmark "${profile_id}" || exit
  done
done
# if the script is interrupted, remember to kill dcgm recorder manually!
kill $pid