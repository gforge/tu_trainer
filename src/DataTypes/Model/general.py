from typing import List, Union
from typing_extensions import Annotated
from pydantic import fields
from pydantic.types import StringConstraints

from DataTypes.enums import OptimizerType
from DataTypes.Network.Cascade.raw import Network_Cfg_Cascade_Raw
from DataTypes.Network.fork import Network_Fork_Cfg_Raw
from DataTypes.Network.fully_connected import Network_Cfg_Fully_Connected_Raw
from DataTypes.Network.morph import Network_Cfg_Morph_Raw
from DataTypes.Network.Pre_defined.raw import Network_Cfg_Pre_Defined_Raw

from DataTypes.general import BaseModelWithGet
from .one_to_one import Model_One_to_One_Cfg
from .one_to_many import Model_One_to_Many_Cfg
from .many_to_one import Model_Many_to_One_Cfg

Any_Model_Cfg = Annotated[
    Union[Model_One_to_Many_Cfg, Model_One_to_One_Cfg, Model_Many_to_One_Cfg],
    fields.Field(discriminator="model_type"),
]
Any_Neural_Net = Annotated[
    Union[
        Network_Cfg_Cascade_Raw,
        Network_Fork_Cfg_Raw,
        Network_Cfg_Morph_Raw,
        Network_Cfg_Fully_Connected_Raw,
        Network_Cfg_Pre_Defined_Raw,
    ],
    fields.Field(discriminator="neural_net_type"),
]


class Model_Cfg(BaseModelWithGet):
    model_type: str = ...
    "The type of model"

    heads: List[Annotated[str, StringConstraints(to_lower=True)]] = ...
    "The input modalities"

    tails: List[Annotated[str, StringConstraints(to_lower=True)]] = ...

    optimizer_type: OptimizerType
    "The type of optimizer to use for backprop"

    neural_net_cfgs: Any_Neural_Net
