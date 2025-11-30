from typing import Literal

from DataTypes.enums import Consistency
from DataTypes.general import BaseModelWithGet
from .base import Network_Cfg_Base


class Network_Fork_Cfg_Raw(BaseModelWithGet):
    neural_net_type: Literal['Fork']
    "Network type name, unique identifier"

    consistency: Consistency
    "The dimensionality of the dat"


class Network_Cfg_Fork_Single_Element(Network_Cfg_Base):
    """A single subnetwork in a fork

    Note that the setup is difficult for a general setup but instead we use
    this per branch config object
    """
    neural_net_type: Literal['Fork'] = 'Fork'

    name: str = ...
    "The assigned name from the fork factory"

    output_channels: int = ...
    "The number of output channels"

    @property
    def add_relu(self) -> bool:
        "Add ReLu activation after fork unless this is the top layer"
        return not self.is_head_input_modality

    def __str__(self) -> str:
        return self.name

    def get_id(self) -> str:
        return self.name
