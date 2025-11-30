from abc import ABCMeta
from typing import Generic, Union

from DataTypes import Consistency
from .base_modality import Base_Modality, Modality_Type


class Base_Sequence(Generic[Modality_Type], Base_Modality[Modality_Type], metaclass=ABCMeta):
    """
    We refer to any modality with 1D consistency as Sequence
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def consistency(self) -> Consistency:
        return Consistency.d1

    @property
    def num_channels(self) -> int:
        return self._cfgs.num_channels

    @property
    def width(self) -> Union[int, None]:
        "The width"
        return self._cfgs.width

    def get_tensor_shape(self):
        return [
            self.get_channels(),
            self.get_width(),
        ]

    def get_channels(self):
        return self.num_channels

    def get_width(self):
        return self.width

    def set_channels(self, num_channels: int):
        self._cfgs.num_channels = num_channels

    def set_width(self, width: int):
        self._cfgs.width = width

    def update_data_dimensions(self, cfgs):
        if shape := cfgs.get('tensor_shape'):
            [num_channels, width] = shape
            self.set_channels(num_channels)
            self.set_width(width)
            return

        if nc := cfgs.get('num_channels'):
            self.set_channels(nc)
        if w := cfgs.get('width'):
            self.set_width(w)
