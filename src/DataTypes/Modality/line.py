from typing import Literal

from .csv import Modality_Csv_Column_Prefixes_Cfg


class Modality_Multi_Line(Modality_Csv_Column_Prefixes_Cfg):
    type: Literal['Multi_Line'] = ...
    "The name of the modality"
