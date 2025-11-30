from typing import Dict, List

from DataTypes.Model.general import Any_Model_Cfg
from DataTypes.general import BaseModelWithGet


class Graph_Models_Cfg_Raw(BaseModelWithGet):
    experiment_models: List[str] = ...
    """The modalities defined in the experiment set

    The most common modalities are:
    - image
    - EXPERIMENT_EXPLICIT_CLASSIFICATION_MODALITIES
    - EXPERIMENT_EXPLICIT_REGRESSION_MODALITIES
    - EXPERIMENT_EXPLICIT_PSEUDO_OUTPUT_MODALITIES
    - EXPERIMENT_EXPLICIT_MODALITIES

    Wher the capital modality tokens are replaced with all matching modalities
    """

    graph_specific_models: Dict[str, Any_Model_Cfg] = ...
    "Specific network models with heads and tails"
