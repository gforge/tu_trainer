from typing import Generic
from .Base_Modalities.base_implicit import Base_Implicit
from .Base_Modalities.base_sequence import Base_Sequence, Modality_Type


class Implicit_Sequence(Generic[Modality_Type], Base_Implicit, Base_Sequence[Modality_Type]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
