from typing import Dict, List

from DataTypes.general import BaseModelWithGet
from DataTypes.Modality.cfg_groups import Any_Graph_Modality_Cfg


class Graph_Modalities_Cfg_Raw(BaseModelWithGet):
    experiment_modalities: List[str]
    """The modalities defined in the experiment set

    The most common modalities are:
    - image
    - EXPERIMENT_EXPLICIT_CLASSIFICATION_MODALITIES
    - EXPERIMENT_EXPLICIT_REGRESSION_MODALITIES
    - EXPERIMENT_EXPLICIT_PSEUDO_OUTPUT_MODALITIES
    - EXPERIMENT_EXPLICIT_MODALITIES

    Wher the capital modality tokens are replaced with all matching modalities
    """

    graph_specific_modalities: Dict[str, Any_Graph_Modality_Cfg]
    "The modalities used within the graph, e.g. style, forks"
