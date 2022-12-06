#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 11/14/2020
PyTorch model runner.
References:
    PyTorch Request Batching Server:
        https://github.com/deep-learning-with-pytorch/dlwpt-code/blob/master/p3ch15/request_batching_server.py
"""

import asyncio
import bisect
import time

import numpy as np
import torch
import torch.hub

from utils.logger import Logger
# we only run 1 inference run at any time (one could schedule between several runners if desired)
from utils.model_hub import load_pytorch_model
from utils.pipeline_manager import PostProcessor, PreProcessor


class Task(object):
    def __init__(self, input_size, done_event, inputs, loop_time):
        self.input_size = input_size
        self.done_event = done_event
        self.inputs = inputs
        self.loop_time = loop_time
        self.output = None
        self.inference_time = None
        self.preprocessing_time = None
        self.postprocessing_time = None

    def __lt__(self, other):
        return self.input_size < other.input_size


class HandlingError(Exception):
    def __init__(self, msg, code=500):
        super().__init__()
        self.handling_code = code
        self.handling_msg = msg


class ModelRunner(object):
    def __init__(
            self,
            model_name, task: str, device, max_batch_size=1, max_wait=0.1, max_queue_size=200,
            server_preprocessing=True,
            share_memory=False,
            loop=None, group_batching=False
    ):
        self.model_name = model_name
        self.task = task.lower()
        self.share_memory = share_memory
        self.max_queue_size = max_queue_size
        self.max_wait = max_wait
        self.max_batch_size = max_batch_size
        self.group_batching = group_batching

        if server_preprocessing:
            self.preprocessor = PreProcessor.get_preprocessor(self.task, model_name=self.model_name)
        else:
            self.preprocessor = PreProcessor.default_preprocessor
        self.postprocessor = PostProcessor.get_postprocessor(self.task)

        # set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # load model
        self.model = load_pytorch_model(model_name=self.model_name, task=self.task).to(self.device).eval()
        if self.share_memory:
            self.model.share_memory()

        self._loop = loop or asyncio.get_event_loop()

        self.queue = []
        self.queue_lock = asyncio.Lock(loop=self._loop)
        self.needs_processing = asyncio.Event(loop=self._loop)
        self.needs_processing_timer = None

        self._logger = Logger(f'Model Runner, {model_name}')
        self._model_runner_task = self._loop.create_task(self.model_runner())

    def terminate(self):
        self._model_runner_task.cancel()

    def schedule_processing_if_needed(self):
        if len(self.queue) >= self.max_batch_size:
            self._logger.debug('next batch ready when processing a batch')
            self.needs_processing.set()
        elif self.queue:
            if self.needs_processing_timer is not None:
                # don't need to schedule a set `need_processing`
                return
            self._logger.debug('queue nonempty when processing a batch, setting next timer')
            self.needs_processing_timer = self._loop.call_at(
                min(x.loop_time for x in self.queue) + self.max_wait,
                self.needs_processing.set
            )

    async def process_input(self, inputs):
        if self.task == 'modelForSequenceClassification':
            our_task = Task(
                input_size=np.prod([len(x) for x in inputs]),
                done_event=asyncio.Event(loop=self._loop),
                inputs=inputs,
                loop_time=self._loop.time()
            )
        elif self.task == 'image_classification':
            our_task = Task(
                input_size=len(inputs),
                done_event=asyncio.Event(loop=self._loop),
                inputs=inputs,
                loop_time=self._loop.time()
            )
        else:
            raise NotImplementedError(f'We do not support task={self.task} so far.')

        async with self.queue_lock:
            if len(self.queue) >= self.max_queue_size:
                raise HandlingError("I'm too busy", code=503)
            if self.group_batching:
                bisect.insort(self.queue, our_task)
            else:
                self.queue.append(our_task)
            self._logger.debug("enqueued task. new queue size {}".format(len(self.queue)))
            self.schedule_processing_if_needed()

        await our_task.done_event.wait()

        handle_times = [
            our_task.preprocessing_time,
            our_task.inference_time,
            our_task.postprocessing_time,
        ]
        return our_task.output, handle_times

    def predict(self, batch: torch.Tensor):
        tick = time.time()
        batch = batch.to(self.device)
        result = self.model(batch)
        return result, time.time() - tick

    async def model_runner(self):
        self._logger.info('started model runner for {}'.format(self.model_name))

        try:
            while True:
                await self.needs_processing.wait()
                self.needs_processing.clear()
                if self.needs_processing_timer is not None:
                    self.needs_processing_timer.cancel()
                    self.needs_processing_timer = None

                async with self.queue_lock:
                    longest_wait = self._loop.time() - min(x.loop_time for x in self.queue)
                    self._logger.debug(
                        f'launching processing. queue size: {len(self.queue)}. longest wait: {longest_wait}')
                    to_process = self.queue[:self.max_batch_size]
                    del self.queue[:len(to_process)]
                    self.schedule_processing_if_needed()

                # so here we copy, it would be neater to avoid this
                preprocessing_start_time = time.time()
                raw_batch = [t.inputs for t in to_process]
                batch = self.preprocessor(raw_batch)
                preprocessing_time = time.time() - preprocessing_start_time

                result, inference_time = await self._loop.run_in_executor(None, self.predict, batch)

                post_processing_start_time = time.time()
                result = self.postprocessor(result)
                post_processing_time = time.time() - post_processing_start_time

                for t, r in zip(to_process, result):
                    t.output = r
                    t.preprocessing_time = preprocessing_time
                    t.inference_time = inference_time
                    t.postprocessing_time = post_processing_time
                    t.done_event.set()
                del to_process
        except asyncio.CancelledError:
            pass
