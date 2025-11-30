from typing import Union
from DataTypes import Modality_Style_Cfg
from .Base_Modalities.base_style import Base_Style
from .Base_Modalities.base_plane import Base_Plane


class Style_Plane(Base_Style, Base_Plane[Modality_Style_Cfg, Union[int, None]]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
