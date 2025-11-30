import torch
import numpy as np


# Helpers for dim bug
def check_dims(var):
    if (not isinstance(var, torch.Tensor) and not isinstance(var, np.ndarray)):
        return False

    if var.shape == ():
        return False

    return True


def fix_dims(var):
    if check_dims(var):
        return var
    return np.array([var])
