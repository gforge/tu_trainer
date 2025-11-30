from typing import Literal

from .csv import Modality_Csv_Mutliple_Columns_Cfg


class Modality_Multi_Bipolar_Cfg(Modality_Csv_Mutliple_Columns_Cfg):
    type: Literal['Multi_Bipolar'] = ...
    "The name of the modality"

    calculate_entropy: bool = False
    """Calculated entropy and add to runtime values, useful for identifying images for closer inspection.
    Not really that useful - dropped by default for speed.
    """

    skip_dictionary_save: bool = False
    "Skip saving dictionary - used during tests"
