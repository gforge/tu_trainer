from abc import ABCMeta
from typing import Generic, TypeVar

from DataTypes import Consistency, Plane_Modalities
from .base_modality import Base_Modality

# As Style has optional width/height we need to be a little flexibles
# with a generic dimension
DimVar = TypeVar('DimVar', int, None)


class Base_Plane(Generic[Plane_Modalities, DimVar], Base_Modality[Plane_Modalities], metaclass=ABCMeta):
    """We refer to any modality with 2D consistency as Plane

    Args:
        Generic (modality, dim): The modality is the config while the dim is the dimentionality of
                                 width, height
        Base_Modality (modality): The inherited modality
        metaclass: ABC - avbstract clas
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def consistency(self) -> Consistency:
        return Consistency.d2

    @property
    def num_channels(self) -> int:
        return self._cfgs.num_channels

    @property
    def height(self) -> DimVar:
        "The height of the input size"
        return self._cfgs.height

    @property
    def width(self) -> DimVar:
        "The width of the input size"
        return self._cfgs.width

    def get_tensor_shape(self):
        return [
            self.get_channels(),
            self.get_width(),
            self.get_height(),
        ]

    def get_channels(self):
        return self.num_channels

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def set_channels(self, num_channels):
        self._cfgs.num_channels = num_channels

    def set_width(self, width):
        self._cfgs.width = width

    def set_height(self, height):
        self._cfgs.height = height

    def update_data_dimensions(self, cfgs):
        if shape := cfgs.get('tensor_shape'):
            [num_channels, height, width] = shape
            self.set_channels(num_channels)
            self.set_width(width)
            self.set_height(height)
            return

        if nc := cfgs.get('num_channels'):
            self.set_channels(nc)
        if w := cfgs.get('width'):
            self.set_width(w)
        if h := cfgs.get('height'):
            self.set_height(h)
