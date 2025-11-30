from typing import Any, Optional
from pydantic import Field
from pydantic.types import PositiveInt

from DataTypes.learning_rate_cfg import Learning_Rate_Cfg
from DataTypes.Scene.extended import Scene_Cfg_Extended
from DataTypes.enums import JitterPool, OptimizerType

from .raw import Task_Cfg_Raw


class Task_Cfg_Extended(Task_Cfg_Raw):
    name: str
    "The name of the task"

    graph_name: Optional[str] = Field(default=None)
    "The graph name"

    scene_name: str = ...
    "The name of the scene"

    scenario_name: Optional[str] = Field(default=None)
    "The name of the scenario"

    jitter_pool: Optional[JitterPool] = Field(default=None)
    "How to pool jittered images"

    view_pool: Optional[str] = Field(default=None)
    "How to merge the views when outcomes are pooled on exam level"

    optimizer_type: Optional[OptimizerType] = Field(default=None)
    "The type of optimizer to deploy"

    has_pseudo_labels: Optional[bool] = Field(default=None)
    "Should pseudo labels be activated"

    pseudo_loss_factor: Optional[float] = Field(default=None)
    "How much pseudo loss from fictive labels compares to regular loss from manually annotated images"

    learning_rate: Optional[Learning_Rate_Cfg] = Field(default=None)
    "The learning rate, usually defined in the scene"

    min_channels: Optional[PositiveInt] = Field(default=None)
    "The minimum number of channels to use for the implicit modalities"

    batch_size: Optional[int] = Field(default=None)
    "The batch size"

    backprop_every: Optional[PositiveInt] = Field(default=None)
    """When to backprop in training

    If we have small batches due to memory limitation we may want to run backprop less than every iteration
    so that we in practice get larger batch size and a more stable training."""

    def __init__(self, **data: Any) -> None:
        scene_cfgs: Scene_Cfg_Extended = data['scene_cfgs']
        del data['scene_cfgs']
        data['scene_name'] = scene_cfgs.name

        super().__init__(**data)

        self._extend_with_another_models_data(scene_cfgs)

        if self.backprop_every is None:
            self.backprop_every = 1
