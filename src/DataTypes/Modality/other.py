from __future__ import annotations
from typing import Any, List, Literal, Optional, Union, TYPE_CHECKING
from typing_extensions import Annotated

from pydantic.types import PositiveInt
from pydantic.fields import Field

from DataTypes.Network.Cascade.raw import Network_Cfg_Cascade_Raw
from DataTypes.Network.Pre_defined.raw import Network_Cfg_Pre_Defined_Raw
if TYPE_CHECKING:
    from DataTypes.Task.extended import Task_Cfg_Extended
from DataTypes.general import BaseModelWithGet
from DataTypes.enums import Consistency

Neural_Net_Cfgs = Annotated[Union[Network_Cfg_Cascade_Raw, Network_Cfg_Pre_Defined_Raw],
                            Field(discriminator="neural_net_type")]


class Modality_Model_Cfgs(BaseModelWithGet):
    "Usually this is a cascade network config with a minimal cascade network"
    model_type: Literal['One_to_One'] = 'One_to_One'
    heads: List[str] = []
    tails: List[str] = []
    neural_net_cfgs: Neural_Net_Cfgs


class Modality_Cfg_Base(BaseModelWithGet):
    type: Literal['Base_Config'] = ...
    "The modality type"

    consistency: Optional[Consistency] = Field(default=None)
    "The consistency for this modality"

    num_channels: Optional[PositiveInt] = Field(default=None)
    "The number of channels"

    tensor_shape: Optional[List[int]] = Field(default=None)
    "The shape of the tensor"

    num_jitters: Optional[PositiveInt] = Field(default=None)
    "The number of jitters to each image"

    height: Optional[PositiveInt] = Field(default=None)

    width: Optional[PositiveInt] = Field(default=None)

    depth: Optional[PositiveInt] = Field(default=None)

    model_cfgs: Optional[Modality_Model_Cfgs] = Field(default=None)
    "Neural net configuration"
