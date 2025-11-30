import os
from collections.abc import Iterable
from typing import Any, Dict, Tuple, TypeVar, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate

gpu_activated = False
if ('DEVICE_BACKEND' in os.environ and os.environ['DEVICE_BACKEND'].lower() in ['cuda', 'gpu']):
    gpu_activated = True

PossiblyTensor = TypeVar('PossiblyTensor')


def move_2_cuda_if_gpu_activated(data: PossiblyTensor) -> PossiblyTensor:
    if isinstance(data, Tensor) and gpu_activated:
        return data.cuda()

    return data


def wrap(data):
    assert(isinstance(data, np.ndarray)),\
        'unknown modality data type %s' % (type(data))
    data_type = data.dtype.name
    if data_type.startswith('object'):
        # TODO: Check when this stack occurs as it will most likely not be handled by collate
        data = np.stack(data)
        data_type = data.dtype.name

    if data_type.startswith('float'):
        data = torch.FloatTensor(data)
    elif data_type.startswith('int'):
        data = torch.LongTensor(data)

    else:
        raise TypeError(f'Unknown numpy data type: "{data_type}"')

    data = move_2_cuda_if_gpu_activated(data)
    return data


def unwrap(data: Union[np.ndarray, Tensor], clone=False) -> np.ndarray:
    # Already in numpy format
    if isinstance(data, np.ndarray):
        if clone:
            data = data.copy()
        return data

    assert isinstance(data, Tensor),\
        'unknown modality data type %s' % (type(data))

    data = data.detach()

    # if os.environ['DEVICE_BACKEND'].lower() in ['cuda'.lower(), 'gpu'.lower()]:
    if data.is_cuda:
        data = data.cpu()
    if clone:
        data = data.clone()

    data = data.numpy()
    return data


IndexeableShape = TypeVar('IndexeableShape', Tuple[int, int], np.ndarray)


def append_shape(destination: Dict[str, Any], shape: IndexeableShape, main_idx: int):
    """
    Used in converting to collating format
    """
    for idx, value in enumerate(shape):
        destination[main_idx, idx] = value

    return destination


def extract_and_set_shape(destination: Dict[str, Any], source_array: Union[Tensor, np.ndarray], name: str,
                          main_idx: int):
    """
    Used in converting from collating format
    """
    shape = source_array[main_idx]
    if isinstance(shape, Tensor):
        shape = shape.detach().cpu().numpy()
    shape = shape[~np.isnan(shape)]

    destination[name] = shape

    return destination


def collate_factory(keys_2_ignore: Tuple[str]):
    r"""
    Returns a function for use with DataLoader that:
    1. puts each data field into a tensor with outer dimension batch size
    2. moves tensor to cuda if DEVICE_BACKEND is CUDA
    3. ignores any dictionary keys specified in the list

    Based on PyTorch's collate function with a minor tweak to dict input
    """

    def collate(batch):
        r"""Puts each data field into a tensor with outer dimension batch size"""
        assert len(batch) == 1, 'A batch should have the length of 1 as it is in dictionary format'
        batch_dictionary = batch[0]
        assert isinstance(batch_dictionary, dict)

        ret = {}

        # The hooks change the keys and therefore we want to do the first sweep here befor running the core loop
        for hook in batch_dictionary['post_batch_hooks']:
            hook(batch=batch_dictionary)
        del batch_dictionary['post_batch_hooks']

        for key in batch_dictionary:
            val = batch_dictionary[key]
            if key in keys_2_ignore:
                ret[key] = val
            elif not isinstance(val, Iterable):
                ret[key] = default_collate([val])[0]
            else:
                ret[key] = default_collate(val)

        return ret

    return collate
