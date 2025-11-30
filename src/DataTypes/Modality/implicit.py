from typing import Literal

from DataTypes.enums import Consistency
from .other import Modality_Cfg_Base


class Modality_Implicit_Cfg(Modality_Cfg_Base):
    type: Literal['Implicit']

    consistency: Consistency
    "The consistency for this modality"

    has_reconstruction_loss: bool = False
    "?Not sure if this is used"
