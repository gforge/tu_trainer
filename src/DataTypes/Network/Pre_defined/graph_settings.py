from DataTypes.general import BaseModelWithGet

from .other import PredDefinedRepoOrDir, PreDefinedModel


class Network_Graph_PreDefiend_Settings(BaseModelWithGet):
    """Extra settings for pre-defined graphs"""

    repo_or_dir: PredDefinedRepoOrDir
    "The name of the repository"

    model: PreDefinedModel
    "The name of the model"

    pretrained: bool
    """Should we download the model and use a pre-trained version or should we use
    the network design and train from scratch?
    """
