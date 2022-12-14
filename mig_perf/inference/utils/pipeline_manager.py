# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 12/8/2020
"""
import io
from typing import Callable

import cv2
import numpy as np
import torch.hub
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from .misc import camelcase_to_snakecase


class PreProcessor(object):
    """
    Class for data pre-processing.
    """

    _tokenizer_dict = dict()

    _image_transform = transforms.Compose([
        io.BytesIO,
        Image.open,
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
    ])

    @staticmethod
    def get_preprocessor(task: str, model_name: str = None, **kwargs):
        """
        processing data here.
        """
        if task == 'image_classification':
            return PreProcessor.transform_image2torch
        elif task == 'single_label_classification':
            return PreProcessor.sequence_classification_preprocessor_factory(model_name, **kwargs)
        else:
            return PreProcessor.default_preprocessor

    @staticmethod
    def default_preprocessor(raw_batch):
        return torch.stack([torch.from_numpy(i) for i in raw_batch])

    @staticmethod
    def resize_image(image, width, height, data_type: str = None):
        if data_type is None:
            return cv2.resize(image, (width, height))
        if data_type == 'float32':
            data_type = np.float32
        elif data_type == 'uint8':
            data_type = np.uint8
        return cv2.resize(image, (width, height)).astype(data_type)

    @classmethod
    def transform_image2torch(cls, images):
        return torch.stack([cls._image_transform(image) for image in images], dim=0)

    @classmethod
    def sequence_classification_preprocessor_factory(cls, model_name, **kwargs):
        tokenizer = cls._tokenizer_dict.get(model_name, None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        return lambda input_str_list: tokenizer(input_str_list, return_tensors='pt', **kwargs)['input_ids']


class PostProcessor(object):
    """
    Class for data post-processing.
    """

    @staticmethod
    def image_classification_postprocessor(outputs):
        return outputs.cpu().detach().tolist()

    @staticmethod
    def nlp_postprocessor(outputs):
        return outputs[0].cpu().detach().tolist()

    @staticmethod
    def sequence_classification_postprocessor(outputs):
        return outputs[0].cpu().detach().tolist()

    @staticmethod
    def get_postprocessor(task: str) -> Callable:
        task = camelcase_to_snakecase(task)
        postprocessor = getattr(PostProcessor, f'{task}_postprocessor', None)
        if postprocessor is None:
            raise ValueError(f'post processor not found for task {task}.')

        return postprocessor
