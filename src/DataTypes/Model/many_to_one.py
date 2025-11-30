from typing import List, Literal
from pydantic import fields

from DataTypes.general import BaseModelWithGet
from DataTypes.Network.morph import Network_Cfg_Morph_Raw


class Model_Many_to_One_Cfg(BaseModelWithGet):
    model_type: Literal['Many_to_One']

    heads: List[str] = fields.Field(...,
                                    title="The heads to the model",
                                    description="The many-to-one can have more than 1 head",
                                    min_items=1)
    "The name of the input node(s)"

    tails: List[str] = fields.Field(...,
                                    title="The tail of the model",
                                    description="A fork can have more than 1 tail",
                                    min_items=1,
                                    max_items=1)
    "The tails of the model, where output goes"

    neural_net_cfgs: Network_Cfg_Morph_Raw = ...
    "A neural network"
