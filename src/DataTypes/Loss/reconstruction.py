from typing import Any, List, Literal
import numpy as np
import torch
from pydantic import ConfigDict, fields

from pydantic.types import PositiveFloat, PositiveInt

from GeneralHelpers.pytorch_wrapper import wrap

from .other import Loss_Cfg_Base


class Loss_L2_Cfgs(Loss_Cfg_Base):
    "Calculate the mse_loss"

    loss_type: Literal['l2_loss'] = ...

    relu: bool = True
    """Whether to use relu activation, defautls to True

    Adds the `mse_loss(relu(output), relu(target))`
    """

    tensor_shape: List[int] = ...


class Loss_L1_Laplacian_Pyramid_Cfgs(Loss_Cfg_Base):
    loss_type: Literal['l1_laplacian_pyramid_loss'] = ...

    num_channels: PositiveInt = ...
    "The number of channels to use"

    pyramid_levels: PositiveInt
    "?"

    sigma: PositiveFloat
    "?"

    kernel_size: PositiveInt
    "The size of the kernel used"

    kernel: torch.Tensor = fields.Field(default=None)
    "A tensor with the kernel"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        self.__init_kernel()

    def __init_kernel(self):
        assert self.kernel_size % 2 == 1, f'Kernel size should be odd but it is {self.kernel_size}'
        grid = np.float32(np.mgrid[0:self.kernel_size, 0:self.kernel_size].T)

        def gaussian(x):
            return np.exp((x - self.kernel_size // 2)**2 / (-2 * self.sigma**2))**2

        kernel = np.sum(gaussian(grid), axis=2)
        kernel /= np.sum(kernel)
        kernel = np.tile(kernel, (self.num_channels, 1, 1))
        self.kernel = wrap(kernel[:, None, :, :])
