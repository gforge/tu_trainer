from typing import Dict, List, Literal, Optional, Union

from pydantic import Field

from DataTypes.general import BaseModelWithGet
from .inheritance import Dataset_Modality_Cfg_Inheritance
from .other import Modality_Cfg_Base


class Dataset_Modality_Column(BaseModelWithGet):
    name: str
    csv_name: str


class Dataset_Modality_Shared_Csv_Cfgs(Modality_Cfg_Base, Dataset_Modality_Cfg_Inheritance):
    modality: Union[Literal['input'], Literal['output']] = ...
    "Input or output modality"

    dictionary: Optional[Dict[str, Union[str, List[str]]]] = Field(default=None)
    "If we want to convert values into alternative values, e.g. postive = right, negative = left"

    to_each_view_its_own_label: bool = False

    ignore_index: Optional[int] = Field(default=None)
    "The value/index to ignore, i.e. missing value"

    signal_to_noise_ratio: float = 1.
    "Set in the dataset config"


class Modality_Csv_Single_Column_Cfg(Dataset_Modality_Shared_Csv_Cfgs):
    type: Literal['One_vs_Rest', 'Hierarchical_Label'] = ...
    "The name of the modality"

    column_name: Union[str, Dataset_Modality_Column]


class Modality_Csv_Mutliple_Columns_Cfg(Dataset_Modality_Shared_Csv_Cfgs):
    type: Literal['Csv_Mutliple_Columns'] = ...
    "The name of the modality"

    columns: List[Union[str, Dataset_Modality_Column]]


class Modality_Csv_Column_Prefixes_Cfg(Dataset_Modality_Shared_Csv_Cfgs):
    type: Literal['Csv_Column_Prefixes'] = ...
    "The name of the modality"

    column_prefixes: List[str]
