import numpy as np

model_names = {
    'distil_v1': 'sentence-transformers/distiluse-base-multilingual-cased-v1',
    'distil_v2': 'sentence-transformers/distiluse-base-multilingual-cased-v2',
    'MiniLM': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'mpnet': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'bert-base': 'bert-base-multilingual-cased',
}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def p99_latency(latency_list: list):
    p99_number = max(1, int(len(latency_list)*0.99))
    latency_list.sort()
    latency = np.mean(latency_list[-p99_number:])
    return latency
