from typing import Literal

from .csv import Modality_Csv_Column_Prefixes_Cfg


class Modality_Multi_Coordinate(Modality_Csv_Column_Prefixes_Cfg):
    type: Literal['Multi_Coordinate'] = ...
    "The name of the modality"
