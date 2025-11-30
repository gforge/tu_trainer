from typing import Literal

from .csv import Modality_Csv_Single_Column_Cfg


class Modality_Bipolar_Cfg(Modality_Csv_Single_Column_Cfg):
    type: Literal['Bipolar'] = ...
    "The name of the modality"

    calculate_entropy: bool = False
    """Calculated entropy and add to runtime values, useful for identifying images for closer inspection.
    Not really that useful - dropped by default for speed.
    """
