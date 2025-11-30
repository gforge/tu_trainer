from torch import Tensor
from .abstract_view_pool import Abstract_View_Pool


class View_Pool_Max(Abstract_View_Pool):

    def __init__(self, keepdim=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keepdim = keepdim

    def forward_per_view(self, x: Tensor):
        # The max() returns a tuple and hence the [0] for selecting not the index but the actual value
        return x.max(dim=self.dim, keepdim=self.keepdim)[0]
