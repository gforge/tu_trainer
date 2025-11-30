from typing import Literal

from pydantic.types import PositiveInt

from DataTypes.enums import Consistency
from .other import Modality_Cfg_Base


class Modality_Style_Cfg(Modality_Cfg_Base):
    type: Literal['Style']

    consistency: Consistency = ...
    "The consistency for this modality"

    distribution: Literal['gaussian'] = ...
    "The distribution of the style"

    num_channels: PositiveInt = ...
    "The number of channels the style uses"

    modality: Literal['output'] = 'output'
    "All styles are by default edges"

    mean: float = 0
    "The mean of the distribition"

    std: float = 1
    "The standard deviation of the distribution"
