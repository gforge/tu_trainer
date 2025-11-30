from typing import Literal, Optional
from DataTypes.general import BaseModel

from .other import PredDefinedRepoOrDir, PreDefinedModel


class Network_Cfg_Pre_Defined_Raw(BaseModel):
    neural_net_type: Literal['Pre-defined']

    repo_or_dir: Optional[PredDefinedRepoOrDir]
    "The name of the repository"

    model: Optional[PreDefinedModel]
    "The name of the model"

    pretrained: Optional[bool]
    """Should we download the model and use a pre-trained version or should we use
    the network design and train from scratch?
    """
