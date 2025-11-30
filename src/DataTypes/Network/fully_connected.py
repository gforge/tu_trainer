from typing import List, Literal

from DataTypes.general import BaseModelWithGet
from DataTypes.enums import Consistency

from .base import Network_Cfg_Base


class Network_Cfg_Fully_Connected_Raw(BaseModelWithGet):
    neural_net_type: Literal['Fully_Connected']

    num_hidden: int = ...
    "The number of hidden units"

    consistency: Consistency = ...
    "The dimensionality of the input data, 1D, 2D or 3D"


class Network_Cfg_Fully_Connected(Network_Cfg_Base, Network_Cfg_Fully_Connected_Raw):
    head_shape: List[int] = ...
    "The input shape"

    tail_shape: List[int] = ...
    "The output shape"

    @property
    def add_relu(self) -> bool:
        "Add ReLu activation after fork unless this is the top layer"
        return not self.is_head_input_modality

    def __str__(self) -> str:
        return self.get_id()

    def get_id(self) -> str:
        head_shape_txt = 'x'.join(str(i) for i in self.head_shape)
        tail_shape_txt = 'x'.join(str(o) for o in self.tail_shape)
        return f'fc_{self.head_name}_{head_shape_txt}_{self.tail_name}_{tail_shape_txt}_{self.purpose}'
