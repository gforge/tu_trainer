import numpy as np
import torch


def init_nan_array(shape, dtype=float):
    arr = np.ndarray(list(shape), dtype=dtype)
    arr[:] = np.nan  # Just precausion so that no initial values are used
    return arr


def init_torch_nan_array(shape, template_tensor: torch.Tensor) -> torch.Tensor:
    return torch.empty(*shape, dtype=template_tensor.dtype, device=template_tensor.device).fill_(float('NaN'))
