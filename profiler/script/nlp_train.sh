batch_sizes=(8 16 32 64 128 256)
sequence_lengths=(32 128 256 512) # seq_length 64 is already tested in batch_size comparison testing

function seq_length_benchmark {
  for seq_length in ${sequence_lengths[*]}
  do
      CUDA_VISIBLE_DEVICES=${1} python A100-nlp-train_test.py \
      "model_name=${2}" "mig_profile=${3}" "seq_length=${seq_length}" "gpu=${4}"\
      "result_dir=${5}"
  done
  return 0
}
export PYTHONPATH="${PWD}"

function batch_size_benchmark {
  for batch_size in ${batch_sizes[*]}
  do
      CUDA_VISIBLE_DEVICES=${1} python A100-nlp-train_test.py \
      "model_name=${2}" "mig_profile=${3}" "batch_size=${batch_size}" "gpu=${4}"\
      "result_dir=${5}"
  done
  return 0
}

batch_size_benchmark "$1" "$2" "$3" "$4" "$5"
seq_length_benchmark "$1" "$2" "$3" "$4" "$5"