from torch import nn, Tensor, cat
import numpy as np
from typing import Optional, Tuple

from GeneralHelpers.pytorch_wrapper import unwrap


class Abstract_View_Pool(nn.Module):
    """
    We want to build our sequential model as far as we can and we need to be able
    to add even trivial steps
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor, num_views: Optional[Tuple] = None):
        if num_views is None:
            return self.forward_per_view(x)

        exam_indexes = np.cumsum([0, *unwrap(num_views)])
        vals = [
            self.forward_per_view(x[:, exam_indexes[i]:exam_indexes[i + 1], :]) for i in range(len(exam_indexes) - 1)
        ]

        return cat(vals, dim=self.dim)

    def forward_per_view(self, x):
        raise NotImplementedError('The forward_per_view is not implemented!')
