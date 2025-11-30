import numpy as np
from torch import Tensor
from .abstract_view_pool import Abstract_View_Pool


class View_Pool_AvgTopProp(Abstract_View_Pool):
    """
    Takes the top proportion and calculates the mean for those.
    The max function automatically ignores the below-max images and
    thus the inference is basically reduced to a single images which
    may be less desireable.
    """

    def __init__(self, prop: float, keepdim=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prop = prop
        self.keepdim = keepdim

    def forward_per_view(self, x: Tensor):
        y, _ = x.sort(dim=self.dim)
        no_imgs = int(np.floor(y.shape[1] * self.prop))
        return y[:, no_imgs:].mean(dim=self.dim, keepdim=self.keepdim)
