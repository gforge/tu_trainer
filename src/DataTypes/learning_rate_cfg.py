from pydantic.types import confloat

from .general import BaseModelWithGet
from .enums import Learning_Rate_Type, Learning_Rate_Function
from pydantic import Field
from typing_extensions import Annotated


class Learning_Rate_Cfg(BaseModelWithGet):
    type: Learning_Rate_Type = Learning_Rate_Type.decay
    "Currently decay learning rate is the only one available"

    function: Learning_Rate_Function = ...
    "How the learning rate should change with each epoch"

    starting_value: Annotated[float, Field(gt=0)]
    "The learning rate to start with, usually 0.025 to 0.2 seem to work"
