from typing import Union, TypeVar

from .csv import Modality_Csv_Single_Column_Cfg, Modality_Csv_Mutliple_Columns_Cfg, \
    Modality_Csv_Column_Prefixes_Cfg
from .bipolar import Modality_Bipolar_Cfg
from .coordinate import Modality_Multi_Coordinate
from .id import Modality_ID_Cfg
from .image import Modality_Image_Cfg
from .implicit import Modality_Implicit_Cfg
from .independent_line import Modality_Multi_Independent_Line
from .line import Modality_Multi_Line
from .multi_bipolar import Modality_Multi_Bipolar_Cfg
from .regression import Modality_Multi_Regression
from .style import Modality_Style_Cfg
from .text import Modality_Text_Cfg

Any_Regression = Union[Modality_Multi_Coordinate, Modality_Multi_Independent_Line, Modality_Multi_Line,
                       Modality_Multi_Regression]
Any_Classification = Union[Modality_Bipolar_Cfg, Modality_Multi_Bipolar_Cfg]
Any_CSV_Modality_Cfgs = Union[Any_Classification, Any_Regression, Modality_Csv_Single_Column_Cfg,
                              Modality_Csv_Mutliple_Columns_Cfg, Modality_Csv_Column_Prefixes_Cfg]
Any_Graph_Modality_Cfg = Union[Modality_Implicit_Cfg, Modality_Style_Cfg]
Any_Modality_Cfg = Union[Any_CSV_Modality_Cfgs, Any_Graph_Modality_Cfg, Modality_ID_Cfg, Modality_Text_Cfg,
                         Modality_Image_Cfg]

# Generic used in modality definitions
Modality_Type = TypeVar('Modality_Type', bound=Any_CSV_Modality_Cfgs)
Plane_Modalities = TypeVar('Plane_Modalities', Modality_Style_Cfg, Modality_Image_Cfg)

CSV_Type = TypeVar('CSV_Type', bound=Any_CSV_Modality_Cfgs)
Classificaiton_Type = TypeVar('Classificaiton_Type', bound=Any_Classification)
Regression_Type = TypeVar('Regression_Type', bound=Any_Regression)
