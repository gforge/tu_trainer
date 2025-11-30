from typing import List, Literal

from DataTypes.general import BaseModelWithGet
from .base import Network_Cfg_Base


class Network_Cfg_Morph_Raw(BaseModelWithGet):
    neural_net_type: Literal['Morph']


class Network_Cfg_Morph_Single_Element(Network_Cfg_Base):
    """A single subnetwork in a morph

    Note that the setup is difficult for a general setup but instead we use
    this per branch config object
    """
    neural_net_type: Literal['Morph'] = 'Morph'

    input_shape: List[int]
    "The expected input shape"

    output_shape: List[int]
    "The expected output shape"

    name: str = ...
    "The assigned name from the morph factory"

    @property
    def add_relu(self) -> bool:
        "Add ReLu activation after fork unless this is the top layer"
        return not self.is_head_input_modality

    def __str__(self) -> str:
        return self.name

    def get_id(self) -> str:
        return self.name
