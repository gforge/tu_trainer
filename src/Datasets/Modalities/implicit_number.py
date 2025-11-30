from typing import Generic
from .Base_Modalities.base_implicit import Base_Implicit
from .Base_Modalities.base_number import Base_Number, Modality_Type


class Implicit_Number(Generic[Modality_Type], Base_Implicit, Base_Number[Modality_Type]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
