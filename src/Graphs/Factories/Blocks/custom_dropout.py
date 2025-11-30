import math
import numpy as np
from torch import nn, Tensor
from DataTypes import Custom_Dropout_Cfg

from global_cfgs import Global_Cfgs


class Custom_Dropout(nn.Module):
    """
    This allows for blocking large sections of the image during
    training, a form of super-dropout where the features cannot depend
    on a small area but must reflect a larger area.
    """

    def __init__(
        self,
        cfgs: Custom_Dropout_Cfg,
    ):
        super().__init__()
        assert cfgs.width > 0, 'Width must be a positive amount of pixels'
        assert cfgs.height > 0, 'Width must be a positive amount of pixels'
        assert cfgs.number >= 0, 'The number of dropouts have to be > 0'
        assert cfgs.prop_border_to_ignore >= 0, 'The border to ignore has to be above or equal to 0'
        assert cfgs.prop_border_to_ignore < 0.5, 'The border to ignore has to be less than 0.5 as we otherwise' + \
            ' skip the entire image'

        self.__cfgs = cfgs

    def forward(self, x: Tensor):
        if not self.training or self.__cfgs.number == 0 or not Global_Cfgs().use_custom_dropout:
            return x

        assert len(x.shape) == 4, 'Expected a 4D input shape'

        height = x.shape[len(x.shape) - 1]
        height_border = math.floor(height * self.__cfgs.prop_border_to_ignore)
        height_start = np.floor(height_border + (np.random.rand(*(*x.shape[0:-2], self.__cfgs.number)) *
                                                 (height - height_border)))

        width = x.shape[len(x.shape) - 2]
        width_border = math.floor(width * self.__cfgs.prop_border_to_ignore)
        width_start = np.floor(width_border + (np.random.rand(*(*x.shape[0:-2], self.__cfgs.number)) *
                                               (width - width_border)))
        width_positions = np.stack((width_start, width_start + self.__cfgs.width), axis=3).clip(0,
                                                                                                width - 1).astype(int)
        height_positions = np.stack((height_start, height_start + self.__cfgs.height),
                                    axis=3).clip(0, height - 1).astype(int)
        out = x.clone()
        for batch_no in range(out.shape[0]):
            for clr_no in range(out.shape[1]):
                for i in range(self.__cfgs.number):
                    ws = width_positions[batch_no, clr_no, i, 0]
                    we = width_positions[batch_no, clr_no, i, 1]
                    hs = height_positions[batch_no, clr_no, i, 0]
                    he = height_positions[batch_no, clr_no, i, 1]
                    out[batch_no, clr_no, ws:we, hs:he] = 0

        # import cv2

        # for i in range(out.shape[0]):
        #     for ii in range(out.shape[1]):
        #         img = out[i, ii].cpu().numpy() * 255
        #         org = x[i, ii].cpu().numpy() * 255
        #         both = np.stack((img, org), axis=1).reshape((img.shape[0], img.shape[0] * 2))
        #         cv2.imwrite(f'/home/max/tmp/test{i}_{ii}.png', both.reshape(*both.shape, 1))

        return out

    @property
    def name(self) -> str:
        return f'Custom_Dropout_{self.__cfgs.number}*{self.__cfgs.width}x{self.__cfgs.height}_{self.__cfgs.prop_border_to_ignore}'
