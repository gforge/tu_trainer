import torch
import torch.nn.functional as F

from DataTypes import Loss_L1_Laplacian_Pyramid_Cfgs
from .base_convergent_loss import Base_Convergent_Loss


class L1_Laplacian_Pyramid_Loss(Base_Convergent_Loss[Loss_L1_Laplacian_Pyramid_Cfgs]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.coef = 1e-1  # Decrease loss to not overpower other losses

    @property
    def num_channels(self) -> int:
        return self._cfgs.num_channels

    @property
    def pyramid_levels(self) -> int:
        return self._cfgs.pyramid_levels

    @property
    def kernel_size(self) -> int:
        return self._cfgs.kernel_size

    @property
    def kernel(self) -> torch.Tensor:
        return self._cfgs.kernel

    @property
    def sigma(self) -> float:
        return self._cfgs.sigma

    def laplacian_pyramid(self, img):
        current = img
        pyramid = []

        for _ in range(self.pyramid_levels):
            filtered = self.conv_gauss(current)
            diff = current - filtered
            pyramid.append(diff)
            current = F.avg_pool2d(filtered, 2)

        pyramid.append(current)
        return pyramid

    def conv_gauss(self, img):
        pad = tuple([self.kernel_size // 2] * 4)
        img = F.pad(img, pad, mode='replicate')
        return F.conv2d(img, self.kernel, groups=self.num_channels)

    def calculate_loss(self, output, target):
        pyramid_output = self.laplacian_pyramid(torch.tanh(output))
        pyramid_target = self.laplacian_pyramid(target.detach())
        return sum(F.l1_loss(o, t) for o, t in zip(pyramid_output, pyramid_target))
