from typing import Literal

from DataTypes.enums import Consistency

from .other import Modality_Cfg_Base


class Modality_Text_Cfg(Modality_Cfg_Base):
    type: Literal['Char_Sequence']

    consistenct: Consistency = Consistency.d1
    "The consistency of text is 1-dimensional"

    column_name: str
    "The column name with the text"

    modality: Literal['input']

    dictionary: str = " .,\\-abcdefghijklmnopqrstuvwxyzäåö$()"
    "The list is split into tokens"

    discard_numbers: bool = True
    "If number should be used"

    sentence_length: int = 256
    "The length of the sentence, should be greater than the standard reports"

    case_sensitive: bool = False
