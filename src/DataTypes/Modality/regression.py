from typing import Literal

from .csv import Modality_Csv_Mutliple_Columns_Cfg


class Modality_Multi_Regression(Modality_Csv_Mutliple_Columns_Cfg):
    type: Literal['Multi_Regression'] = ...
    "The name of the modality"
