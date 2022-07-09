import logging
import subprocess
import time
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModel
import os
import pandas as pd

pd.set_option('display.max_columns', None)

models = {
    'distil_v1': 'sentence-transformers/distiluse-base-multilingual-cased-v1',
    'distil_v2': 'sentence-transformers/distiluse-base-multilingual-cased-v2',
    'MiniLM': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'mpnet': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'bert-base': 'bert-base-multilingual-cased',
}


def tokenize_and_numericalize_data(example, tokenizer):
    ids = tokenizer(example['review_body'], truncation=True)['input_ids']
    return {'ids': ids}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def padding(batch, max_seq_len):
    for i, token in enumerate(tqdm(batch)):
        if len(token) > max_seq_len:
            batch[i] = token[:max_seq_len]
        else:
            batch[i] = token + [0] * (max_seq_len - len(token))
    return torch.LongTensor(batch)


def fixed_time_infer_test(model, batch_data, fixed_time_period=120):
    with torch.no_grad():
        # gpu warm up
        for _ in range(20):
            _ = model(batch_data)
        latency = []
        period = 0
        batch_num = 0
        start_timestamp = int(time.time())
        while period < fixed_time_period:
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(batch_data)
            ender.record()
            torch.cuda.synchronize()
            latency += [starter.elapsed_time(ender)]
            period = time.time() - start_timestamp
            batch_num = batch_num + 1
        end_timestamp = int(time.time())
    throughput = 1000*batch_num * len(batch_data) / np.sum(latency)  # per second
    latency_mean = np.mean(latency)
    latency_std = np.std(latency)
    batch_data = batch_data.to('cpu')
    del batch_data
    torch.cuda.empty_cache()
    return latency_mean, latency_std, throughput, start_timestamp, end_timestamp


def fixed_iter_infer_test(model, batch_data, batch_num=300):
    with torch.no_grad():
        # gpu warm up
        for _ in range(20):
            _ = model(batch_data)
        latency = []
        start_timestamp = int(time.time())
        for batch in range(batch_num):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(batch_data)
            ender.record()
            torch.cuda.synchronize()
            latency += [starter.elapsed_time(ender)]
        end_timestamp = int(time.time())
    throughput = 1000*batch_num * len(batch_data) / np.sum(latency)  # per second
    latency_mean = np.mean(latency)
    latency_std = np.std(latency)
    batch_data = batch_data.to('cpu')
    del batch_data
    torch.cuda.empty_cache()
    return latency_mean, latency_std, throughput, start_timestamp, end_timestamp


def easy_infer_experiment(test_data, model_name, mig_profile_id, fixed_time, experiment_args, result_dir, device):
    # prepare model
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    total_param = count_parameters(model)

    latency_mean_list = []
    latency_std_list = []
    throughput_list = []
    start_timestamp_list = []
    end_timestamp_list = []
    for args in experiment_args:
        # gpu clear and cool down
        torch.cuda.empty_cache()
        time.sleep(10)
        batch = padding(test_data[:args['batch_size']], args['seq_length']).to(device)
        if fixed_time:
            latency_mean, latency_std, throughput, start_timestamp, end_timestamp = fixed_time_infer_test(model, batch)
        else:
            latency_mean, latency_std, throughput, start_timestamp, end_timestamp = fixed_iter_infer_test(model, batch)
        latency_mean_list += [latency_mean]
        latency_std_list += [latency_std]
        throughput_list += [throughput]
        start_timestamp_list += [start_timestamp]
        end_timestamp_list += [end_timestamp]
        batch = batch.to('cpu')
        torch.cuda.empty_cache()
        del batch
    result = pd.DataFrame({
        'model_name': [model_name.split('/')[-1]] * len(experiment_args),
        'total_params': [total_param] * len(experiment_args),
        'seq_length': [args['seq_length'] for args in experiment_args],
        'batch_size': [args['batch_size'] for args in experiment_args],
        'latency': latency_mean_list,
        'latency_std': latency_std_list,
        'throughput': throughput_list,
        'start_timestamp': start_timestamp_list,
        'end_timestamp': end_timestamp_list,
    }).round(2)
    save_file = result_dir + '{}_MIG_profile_{}.csv'.format(model_name.split('/')[-1], mig_profile_id)
    keep_head = False if os.path.exists(save_file) else True
    result.to_csv(save_file, mode='a', index=False, header=keep_head)

    model = model.to('cpu')
    torch.cuda.empty_cache()
    del model


@hydra.main(version_base=None, config_path='configs', config_name='infer')
def main(cfg: DictConfig):
    logger = logging.getLogger(cfg.model_name)
    # experiment hyper-parameters
    result_dir = cfg.result_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    experiment_args = cfg.test_args
    model_name = models[cfg.model_name]
    assert torch.cuda.is_available(), 'cuda is not available'
    device = torch.device('cuda:0')
    try:
        print(cfg.mig_profile_id)
        p = subprocess.Popen(args="python /root/infer_test/dcgm_recorder.py", shell=True)
        logger.info("dcgm recorder process {} started!".format(p.pid))
    except Exception as e:
        logger.warning("dcgm recorder failed: {}".format(e))
    # gpu clear and cool down
    time.sleep(10)
    torch.cuda.empty_cache()
    # prepare test data
    _, test_data = load_dataset("amazon_reviews_multi", "all_languages", split=['train', 'test'])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_data = test_data.map(tokenize_and_numericalize_data, fn_kwargs={'tokenizer': tokenizer})
    test_data = [i['ids'] for i in test_data]

    # do the test
    easy_infer_experiment(
        test_data=test_data,
        model_name=model_name,
        mig_profile_id=cfg.mig_profile_id,
        fixed_time=cfg.fixed_time,
        experiment_args=experiment_args,
        result_dir=result_dir,
        device=device,
    )
    p.terminate()
    logger.info("dcgm process is terminated.")
    logger.info("inference completed.")


if __name__ == "__main__":
    main()
