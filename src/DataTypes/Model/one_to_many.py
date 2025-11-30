from typing import List, Literal
from pydantic import fields

from DataTypes.Network.fork import Network_Fork_Cfg_Raw
from DataTypes.general import BaseModelWithGet


class Model_One_to_Many_Cfg(BaseModelWithGet):
    model_type: Literal['One_to_Many']

    heads: List[str] = fields.Field(...,
                                    title="The head to the model",
                                    description="The one-to-many has only 1 head",
                                    min_items=1,
                                    max_items=1)
    "The name of the input node(s)"

    tails: List[str] = fields.Field(...,
                                    title="The tails of the model",
                                    description="A one-to-many can have more than 1 tail",
                                    min_items=1)
    "The tails of the model, where output goes"

    neural_net_cfgs: Network_Fork_Cfg_Raw = ...
    "A neural network config"
