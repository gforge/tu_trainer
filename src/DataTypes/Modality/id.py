from typing import Literal, Optional
from pydantic import Field
from pydantic.types import PositiveInt

from DataTypes.enums import Consistency
from .other import Modality_Cfg_Base


class Modality_ID_Cfg(Modality_Cfg_Base):
    type: Literal['ID_from_Indices']

    consistency: Consistency

    num_channels: PositiveInt = 128
    "The number of channels to rely on"

    modality: Literal['output']

    ignore_index: Optional[int] = Field(default=None)
    "The NA value, not really that important here but it follows the logic of CSV-files"
