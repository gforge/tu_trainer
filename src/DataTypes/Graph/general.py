from typing import Any, Dict, Optional
from pydantic import Field
from DataTypes.enums import JitterPool, OptimizerType
from DataTypes.general import BaseModelWithGet
from DataTypes.learning_rate_cfg import Learning_Rate_Cfg
from DataTypes.Task.extended import Task_Cfg_Extended
from DataTypes.Task.activators import Task_Cfg_Activators
from DataTypes.Network.Pre_defined.graph_settings import Network_Graph_PreDefiend_Settings
from .modalities import Graph_Modalities_Cfg_Raw
from .models import Graph_Models_Cfg_Raw
from typing_extensions import Annotated


class Graph_Cfg_Raw(BaseModelWithGet):
    min_channels: Annotated[int, Field(ge=2)] = Field(default=2)
    'Minimum representation size for when classes are smaller than a specific dim'

    modalities: Graph_Modalities_Cfg_Raw = ...
    "The modalities internal to the graph"

    models: Graph_Models_Cfg_Raw = ...
    "The network models with heads and tails"


class Graph_Cfg_Extended(Graph_Cfg_Raw):
    name: str = Field(default=None)
    "The graph name (set from task_cfg)"

    dataset_name: Optional[str] = Field(default=None)
    "The dataset name (set from task_cfg)"

    scene_name: Optional[str] = Field(default=None)
    "The dataset name (set from task_cfg)"

    optimizer_type: OptimizerType = Field(default=None)
    "The type of optimizer to deploy"

    apply: Task_Cfg_Activators = Field(default=None)
    "What sections should be activated"

    view_dropout: Optional[Annotated[float, Field(ge=0, lt=1)]] = Field(default=None)
    "Set if we want to drop sections of the view"

    jitter_pool: Optional[JitterPool] = Field(default=None)
    "How to pool jittered images"

    view_pool: Optional[str] = Field(default=None)
    "How to merge the views when outcomes are pooled on exam level"

    has_pseudo_labels: bool = Field(default=None)
    "Should pseudo labels be activated"

    pseudo_loss_factor: float = Field(default=None)
    "How much pseudo loss from fictive labels compares to regular loss from manually annotated images"

    learning_rate: Learning_Rate_Cfg = Field(default=None)
    "The learning rate, usually defined in the scene"

    graph_settings: Optional[Dict[str, Network_Graph_PreDefiend_Settings]] = Field(default=None)
    "Settings for specific graph"

    def __init__(self, **data: Any) -> None:
        task_cfgs: Task_Cfg_Extended = data['task_cfgs']
        del data['task_cfgs']
        data['name'] = task_cfgs.graph_name

        super().__init__(**data)

        self._extend_with_another_models_data(task_cfgs)
