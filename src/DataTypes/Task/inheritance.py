from typing import Dict, Optional
from pydantic import Field

from DataTypes.enums import JitterPool, OptimizerType
from DataTypes.Network.Pre_defined.graph_settings import Network_Graph_PreDefiend_Settings
from DataTypes.Modality.cfg_groups import Any_Modality_Cfg
from DataTypes.Modality.inheritance import Dataset_Modality_Cfg_Inheritance


class Task_Cfg_Inheritance(Dataset_Modality_Cfg_Inheritance):
    """Variables that can be supplemented by configs higher up in the hierarchy
    """
    graph_name: Optional[str] = Field(default=None)
    "The name of the used network graph"

    view_dropout: Optional[float] = Field(ge=0, lt=1, default=None)
    "Set if we want to drop sections of the view"

    jitter_pool: Optional[JitterPool] = Field(default=None)
    "How to pool jittered images"

    view_pool: Optional[str] = Field(default=None)
    "How to merge the views when outcomes are pooled on exam level"

    optimizer_type: Optional[OptimizerType] = Field(default=None)
    "The type of optimizer to deploy"

    has_pseudo_labels: Optional[bool] = Field(default=None)
    "Should pseudo labels be activated"

    graph_settings: Optional[Dict[str, Network_Graph_PreDefiend_Settings]] = Field(default=None)
    "Settings for specific graph"

    experiment_modality_settings: Optional[Dict[str, Any_Modality_Cfg]] = Field(default=None)
    """Settings for specific experiment modalities

    E.g. custom image size is specific for Efficientnet and thus it makes
    sense to define the image input size in the yaml file and not among the
    shared settings for all networks.
    """
