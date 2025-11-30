from abc import ABCMeta
from typing import Generic

from DataTypes import Consistency

from .base_modality import Base_Modality, Modality_Type


class Base_Number(Generic[Modality_Type], Base_Modality[Modality_Type], metaclass=ABCMeta):
    """
    We refer to any modality with 1D consistency as Sequence
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def consistency(self) -> Consistency:
        "A number consistency is always of type number"
        return Consistency.number

    @property
    def num_channels(self) -> int:
        return self._cfgs.num_channels

    def get_tensor_shape(self):
        return [self.get_channels()]

    def get_channels(self):
        return self.num_channels

    def set_channels(self, num_channels):
        self._cfgs.num_channels = num_channels

    def update_data_dimensions(self, cfgs):
        if shape := cfgs.get('tensor_shape'):
            [num_channels] = shape
            self.set_channels(num_channels)
            return

        if nc := cfgs.get('num_channels'):
            self.set_channels(nc)

    def is_regression(self):
        return True
