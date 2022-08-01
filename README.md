# Infer_Finetune_Benchmark
a multilingual models infering benchmark and vision transormer finetune and MoCo resnet50 finetune benchmark on Nvidia A100 device.

## Quick Start for NLP Inference Benchmark

#### Prepare python environments

```shell
conda env create -f environments.yaml
conda activate benchmark
pip install -r requirements.txt
```



#### Prepare DCGM

1. Install Nvidia DCGM as guided [here](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/getting-started.html).

2. Enable the DCGM system service.

   ```shell
   sudo systemctl --now enable nvidia-dcgm
   ```

3. Create groups of devices for DCGM to monitor(add the GPU 0 and the a GPU Instances(using entity ID 0) to the group.

   ```shell
   dcgmi group -c my_group -a 0,i:0
   ```

4. list the devices added to the group and see that the group contains the GPU (GPU:0) and GPU Instance 0. 

   ```shell
   dcgmi group -l
   ```

   ```
   +-------------------+----------------------------------------------------------+
   | GROUPS                                                                       |
   | 1 group found.                                                               |
   +===================+==========================================================+
   | Groups            |                                                          |
   | -> 8              |                                                          |
   |    -> Group ID    | 8                                                        |
   |    -> Group Name  | my_group                                                  |
   |    -> Entities    | GPU 0, GPU_I 0                                           |
   +-------------------+----------------------------------------------------------+
   ```

   

#### Configuration

1. DCGM configuration(necessary)

   ```yaml
   group_id: 8
   instance_id: 0
   save_dir: /Path/to/InferFinetuneBenchmark/dcgm_infer_result_saving_dir
   ```

   Set a correct `group_id` and `instance_id` according to the dcgm group you created above.  Assign a `save_dir` to save the DCGM monitor result file.

2. infer benchmark configuration

   ```yaml
   model_name: distil_v1
   result_dir: root/InferFinetuneBenchmark/data/infer_bsz/distil_v1
   mig_profile_id: 0
   fixed_time: True
   test_args: [
           {'seq_length': 64, 'batch_size': 1},
           {'seq_length': 64, 'batch_size': 2},
           {'seq_length': 64, 'batch_size': 4},
           {'seq_length': 64, 'batch_size': 8},
           {'seq_length': 64, 'batch_size': 16},
           {'seq_length': 64, 'batch_size': 32},
           {'seq_length': 64, 'batch_size': 64},
           {'seq_length': 64, 'batch_size': 128},
           {'seq_length': 64, 'batch_size': 256},
       ]
   dcgm:
     exc_path: /Path/to/InferFinetuneBenchmark/dcgm_recorder.py
     group_id: 8
     instance_id: 0
     save_dir: /Path/to/InferFinetuneBenchmark/dcgm_infer_result_saving_dir/
   hydra:
     run:
       dir: /Path/to/InferFinetuneBenchmark/logs/infer/distil_v1/${now:%Y-%m-%d}/${now:%H-%M-%S}
   ```

   `model_name` : 5 nlp models are supported: 'distil_v1', 'distil_v2', 'MiniLM', 'mpnet' , 'bert-base'

   `result_dir`: results of inference benchmark

   `mig_profile_id`: profile id of MIG devices, we are going to use 

   0, 9, 14, 19 profiles which respectively refers to 40g, 20g, 10g, 5g MIG instances.

   `fixed_time`: decide if we do the every inference in the fixed time period. True means every inference is carried out for 120s, while False means every inference will be carried out for 300 iterations.

   `dcgm.exc_path`: the dcgm monitor script `dcgm_recorder.py` location. 

   `hydra.run.dir`: directory of hydra logger files

#### Run the benchmark script

   ```shell
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
   ```

   The `infer.sh` is for inference experiments of all 5 nlp models on all 4 MIG profiling( 20 experiments in total ). Assign proper values for `result_dir` and `test_args` and `hydra.run.dir` by yourself to correctly run the experiments.

   Then run:

   ```shell
   bash infer.sh
   ```

   

