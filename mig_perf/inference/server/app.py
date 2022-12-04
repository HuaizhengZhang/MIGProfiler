#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: USER
Email: yli056@e.ntu.edu.sg
Date: 7/23/2020
RESTful Endpoints.
"""
import asyncio
import os
import pickle
import time
from functools import partial

import numpy as np
import sanic.response as res
from sanic import Sanic
from sanic.request import Request

from torch_model_runner import ModelRunner
from utils.dtype import deserialize_bytes_tensor, model_data_type_to_np
from utils.logger import Logger

logger = Logger(name='restful', welcome=False)

system_start_time = 0


def as_numpy(infer_request):
    """
    From https://github.com/triton-inference-server/server/blob/796b631bd08f8e48ca4806d814f090636599a8f6/src/clients/python/library/tritonclient/grpc/__init__.py#L1588
    Get the tensor data for input associated with this object in numpy format
    Args:
        infer_request:
    Returns:
        np.array: The numpy array containing the response data for the tensor or
            None if the data for specified tensor name is not found.
    """
    shape = list(infer_request['shape'])
    datatype = infer_request['datatype']
    if datatype == 'TYPE_BYTES':
        # String results contain a 4-byte string length
        # followed by the actual string characters. Hence,
        # need to decode the raw bytes to convert into
        # array elements.
        np_array = deserialize_bytes_tensor(infer_request['raw_input_contents'])
    else:
        np_array = np.frombuffer(
            infer_request['raw_input_contents'],
            dtype=model_data_type_to_np(datatype)
        )
    np_array = np.resize(np_array, shape)

    return np_array


async def predict(request: Request, app):
    """Predict handler. This function should be bind with argument `app` before using as a handler.
    Args:
        request (sanic.request.Request): Sanic HTTP request.
        app (HttpServer): Sanic HTTP Server.
    Returns:
        A JSON object containing:
            response: inference response.
            time: a dictionary of times including latency of all stages
    """
    receive_time = time.time()
    # package "decoding" to ndarray
    request_bytes = request.files.get('content').body
    infer_request = pickle.loads(request_bytes)
    inputs = as_numpy(infer_request)

    process_start_time = time.time()
    result, handle_times = await app.ctx['model_runner'].process_input(inputs)
    process_time = time.time() - process_start_time

    handle_time = {
        'preprocessing_time':  handle_times[0],
        'batching_time': process_time - sum(handle_times),
        'inference_time': handle_times[1],
        'postprocessing_time': handle_times[2],
        'server_end2end_time': time.time() - receive_time,
    }

    return res.json({
        'response': result,
        'times': handle_time,
    })


class HttpServer(Sanic):
    """Sanic HTTP server.
    Args:
        name (str): Sanic application name.
    Attributes:
        model_runner (ModelRunner or None): Model inference job executor.
    References:
        Running an HTTP & GRPC Python Server Concurrently on AWS Elastic Beanstalk:
        https://medium.com/swlh/running-an-http-grpc-python-server-concurrently-on-aws-elastic-beanstalk-8524d15030e5
    """

    def __init__(self, name: str):
        ctx = {
            'model_name': MODEL_NAME,
            'task': TASK,
            'server_preprocessing': SERVER_PREPROCESSING,
            'model_runner': None
        }
        super().__init__(name=name, ctx=ctx)

        self.register_listener(self._notify_before_server_start, 'before_server_start')
        self.register_listener(self._notify_server_start, 'after_server_start')

        self._setup_router()

    def _setup_router(self):
        predict_func = partial(predict, app=self)
        predict_func.__name__ = predict.__name__
        predict_func.__module__ = predict.__module__

        self.add_route(predict_func, uri='/predict', methods=['POST'])

    async def load_init_replicas(self):
        # TODO: get the batching configuration here.
        self.ctx['model_runner'] = ModelRunner(
            model_name=self.ctx['model_name'], task=self.ctx['task'], device=f'cuda',
            server_preprocessing=self.ctx['server_preprocessing'], loop=asyncio.get_event_loop(),
        )

    def _notify_before_server_start(self, *args):
        global system_start_time

        system_start_time = time.time()

    async def _notify_server_start(self, *args):
        await self.load_init_replicas()


if __name__ == '__main__':
    MODEL_NAME = os.getenv('MODEL_NAME')
    TASK = os.getenv('TASK')
    DEVICE_ID = os.getenv('DEVICE_ID')
    PORT = int(os.getenv('PORT', '50075'))
    SERVER_PREPROCESSING = os.getenv('SERVER_PREPROCESSING', '0').upper() in ['1', 'TRUE', 'Y', 'YES']

    # Mask out other cuda devices
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID

    app = HttpServer(name='PyTorch-Inference-Server')
    app.run(host='0.0.0.0', port=PORT)
