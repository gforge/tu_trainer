from torch import Tensor
from .abstract_view_pool import Abstract_View_Pool


class View_Pool_Mean(Abstract_View_Pool):

    def __init__(self, keepdim=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keepdim = keepdim

    def forward_per_view(self, x: Tensor):
        return x.mean(dim=self.dim, keepdim=self.keepdim)
