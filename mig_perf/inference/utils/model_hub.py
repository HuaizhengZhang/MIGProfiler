#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 7/8/2020
Obtain the pre-trained PyTorch model object.
"""

import torch
import torch.hub

# Supported model architecture
_MODEL_CV_REPOSITORY = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
_MODEL_NLP_REPOSITORY = [
    'distilbert-base-cased',
    'bert-base-cased',
    'bert-large-cased',
    'xlnet-base-cased',
    'roberta-base'
]


def load_pytorch_model(model_name: str, task: str = 'model'):
    """
    Load model by model name (and task name).
    Supported model names:
    CV:
        -   'resnet18'
        -   'resnet34'
        -   'resnet50'
        -   'resnet101'
        -   'resnet152'
    NLP:
        -   'distilbert-base-cased'
        -   'bert-base-cased'
        -   'bert-large-cased'
        -   'xlnet-base-cased'
        -   'roberta-base'
    Currently supported task (for NLP only):
    -   'model'
    -   'modelForSequenceClassification'
    Args:
        model_name (str): Model name.
        task (str): Model task.
    """

    if model_name in _MODEL_CV_REPOSITORY:
        import torchvision
        return getattr(torchvision.models, model_name)(pretrained=True)
    elif model_name in _MODEL_NLP_REPOSITORY:
        return torch.hub.load('huggingface/pytorch-transformers', task, model_name)
    else:
        raise ValueError(f'model name={model_name} not supported.')
