from abc import ABCMeta
from typing import Generic, Union

from DataTypes import Consistency
from .base_modality import Base_Modality, Modality_Type


class Base_Volume(Generic[Modality_Type], Base_Modality[Modality_Type], metaclass=ABCMeta):
    """
    We refer to any modality with 3D consistency as Volume
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def consistency(self) -> Consistency:
        return Consistency.d3

    @property
    def num_channels(self) -> int:
        return self._cfgs.num_channels

    @property
    def height(self) -> Union[int, None]:
        "The height of the input size"
        return self._cfgs.height

    @property
    def width(self) -> Union[int, None]:
        "The width of the input size"
        return self._cfgs.width

    @property
    def depth(self) -> Union[int, None]:
        "The width of the input size"
        return self._cfgs.depth

    def get_tensor_shape(self):
        return [
            self.get_channels(),
            self.get_depth(),
            self.get_width(),
            self.get_height(),
        ]

    def get_channels(self):
        return self.num_channels

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_depth(self):
        return self.depth

    def set_channels(self, num_channels):
        self._cfgs.num_channels = num_channels

    def set_width(self, width):
        self._cfgs.width = width

    def set_height(self, height):
        self._cfgs.height = height

    def set_depth(self, depth):
        self._cfgs.depth = depth

    def update_data_dimensions(self, cfgs):
        if shape := cfgs.get('tensor_shape'):
            [num_channels, depth, height, width] = shape
            self.set_channels(num_channels)
            self.set_width(width)
            self.set_height(height)
            self.set_depth(depth)
            return

        if nc := cfgs.get('num_channels'):
            self.set_channels(nc)
        if w := cfgs.get('width'):
            self.set_width(w)
        if h := cfgs.get('height'):
            self.set_height(h)
        if d := cfgs.get('depth'):
            self.set_depth(d)
