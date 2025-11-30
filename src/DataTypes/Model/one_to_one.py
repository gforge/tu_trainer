from typing import List, Literal, Union
from pydantic import fields

from DataTypes.general import BaseModelWithGet
from DataTypes.Network.Pre_defined.raw import Network_Cfg_Pre_Defined_Raw
from DataTypes.Network.Cascade.raw import Network_Cfg_Cascade_Raw


class Model_One_to_One_Cfg(BaseModelWithGet):
    model_type: Literal['One_to_One']

    heads: List[str] = fields.Field(...,
                                    title="The head to the model",
                                    description="The one-toone has only 1 head",
                                    min_items=1,
                                    max_items=1)
    "The name of the input node"

    tails: List[str] = fields.Field(...,
                                    title="The tail of the model",
                                    description="A one-to-one must have only 1 tail",
                                    min_items=1,
                                    max_items=1)
    "The tail of the model, where output goes"

    neural_net_cfgs: Union[Network_Cfg_Cascade_Raw,
                           Network_Cfg_Pre_Defined_Raw] = fields.Field(..., discriminator='neural_net_type')
    "A neural network config"
