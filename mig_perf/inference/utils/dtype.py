#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 7/8/2020
Request data type related utility functions.
"""
import struct
from collections import defaultdict
from enum import Enum

import numpy as np


class DataType(Enum):
    """A simplified version of Triton DataType"""
    TYPE_INVALID = 0
    TYPE_BOOL = 1
    TYPE_UINT8 = 2
    TYPE_UINT16 = 3
    TYPE_UINT32 = 4
    TYPE_UINT64 = 5
    TYPE_INT8 = 6
    TYPE_INT16 = 7
    TYPE_INT32 = 8
    TYPE_INT64 = 9
    TYPE_FP16 = 10
    TYPE_FP32 = 11
    TYPE_FP64 = 12
    TYPE_BYTES = 13


def model_data_type_to_np(model_dtype):
    mapper = {
        DataType.TYPE_INVALID: None,
        DataType.TYPE_BOOL: np.bool,
        DataType.TYPE_UINT8: np.uint8,
        DataType.TYPE_UINT16: np.uint16,
        DataType.TYPE_UINT32: np.uint32,
        DataType.TYPE_UINT64: np.uint64,
        DataType.TYPE_INT8: np.int8,
        DataType.TYPE_INT16: np.int16,
        DataType.TYPE_INT32: np.int32,
        DataType.TYPE_INT64: np.int64,
        DataType.TYPE_FP16: np.float16,
        DataType.TYPE_FP32: np.float32,
        DataType.TYPE_FP64: np.float64,
        DataType.TYPE_BYTES: np.object,
    }

    if isinstance(model_dtype, int):
        model_dtype = DataType(model_dtype)
    elif isinstance(model_dtype, str):
        model_dtype = DataType[model_dtype]
    elif not isinstance(model_dtype, DataType):
        raise TypeError(
            f'model_dtype is expecting one of the type: `int`, `str`, or `DataType` but got {type(model_dtype)}'
        )
    return mapper[model_dtype]


def type_to_data_type(tensor_type: type):
    mapper = defaultdict(
        lambda: DataType.TYPE_INVALID, {
            bool: DataType.TYPE_BOOL,
            int: DataType.TYPE_INT32,
            float: DataType.TYPE_FP32,
            str: DataType.TYPE_BYTES,
            np.dtype(np.bool): DataType.TYPE_BOOL,
            np.dtype(np.uint8): DataType.TYPE_UINT8,
            np.dtype(np.uint16): DataType.TYPE_UINT16,
            np.dtype(np.uint32): DataType.TYPE_UINT32,
            np.dtype(np.uint64): DataType.TYPE_UINT64,
            np.dtype(np.float16): DataType.TYPE_FP16,
            np.dtype(np.float32): DataType.TYPE_FP32,
            np.dtype(np.float64): DataType.TYPE_FP64,
            np.dtype(np.object): DataType.TYPE_BYTES,
        }
    )

    return mapper[tensor_type]


def serialize_byte_tensor(input_tensor: np.array):
    """
    From https://github.com/triton-inference-server/server/blob/796b631bd08f8e48ca4806d814f090636599a8f6/src/clients/python/library/tritonclient/utils/__init__.py
    Serializes a bytes tensor into a flat numpy array of length prepend bytes.
    Can pass bytes tensor as numpy array of bytes with dtype of np.bytes_,
    numpy strings with dtype of np.str_ or python strings with dtype of np.object.
    Args:
        input_tensor (np.array): The bytes tensor to serialize.
    Returns:
        np.array: The 1-D numpy array of type uint8 containing the serialized bytes in 'C' order.
    Raises:
        ValueError: If unable to serialize the given tensor.
    """

    if input_tensor.size == 0:
        return np.empty([0])

    # If the input is a tensor of string/bytes objects, then must flatten those into
    # a 1-dimensional array containing the 4-byte byte size followed by the
    # actual element bytes. All elements are concatenated together in "C"
    # order.
    if (input_tensor.dtype == np.object) or (input_tensor.dtype.type
                                             == np.bytes_):
        flattened = bytes()
        for obj in np.nditer(input_tensor, flags=["refs_ok"], order='C'):
            # If directly passing bytes to BYTES type,
            # don't convert it to str as Python will encode the
            # bytes which may distort the meaning
            if obj.dtype.type == np.bytes_:
                if type(obj.item()) == bytes:
                    s = obj.item()
                else:
                    s = bytes(obj)
            else:
                s = str(obj).encode('utf-8')
            flattened += struct.pack('<I', len(s))
            flattened += s
        flattened_array = np.asarray(flattened)
        if not flattened_array.flags['C_CONTIGUOUS']:
            flattened_array = np.ascontiguousarray(flattened_array)
        return flattened_array
    else:
        raise ValueError('cannot serialize bytes tensor: invalid datatype')


def deserialize_bytes_tensor(encoded_tensor):
    """
    Deserializes an encoded bytes tensor into an
    numpy array of dtype of python objects
    Args:
        encoded_tensor (bytes): The encoded bytes tensor where each element
            has its length in first 4 bytes followed by the content
    Returns:
        np.array: The 1-D numpy array of type object containing the
            deserialized bytes in 'C' order.
    """
    strs = list()
    offset = 0
    val_buf = encoded_tensor
    while offset < len(val_buf):
        l = struct.unpack_from("<I", val_buf, offset)[0]
        offset += 4
        sb = struct.unpack_from("<{}s".format(l), val_buf, offset)[0]
        offset += l
        strs.append(sb)
    return np.array(strs, dtype=bytes)
