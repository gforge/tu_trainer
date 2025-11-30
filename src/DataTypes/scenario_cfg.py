from typing import Any, Optional

from pydantic import Field
from pydantic.types import PositiveInt

from .enums import JitterPool, LossWeightType, OptimizerType
from .Scene.inheritance import Scene_Cfg_Inheritance
from .Scene.raw import Scene_Cfg_Raw


class Scenario_Cfg(Scene_Cfg_Inheritance):
    """
    Scenario_Cfg is a configuration class for defining various parameters and settings for a scenario.

    Attributes:
        name (str): The name of the scenario, provided separately.
        graph_name (str): The name of the graph that we want to be using.
        optimizer_type (OptimizerType): The type of optimizer to deploy.
        main_task (Optional[str]): The name of the task, if ignored it will default to the first task in the scene task list.
        jitter_pool (JitterPool): How to pool jittered images.
        view_pool (str): How to merge the views when outcomes are pooled on exam level.
        scenes (List[Scene_Cfg_Raw]): The scenes that we want to run. Must have at least one scene.
        min_channels (Optional[PositiveInt]): The minimum number of channels to use for the implicit modalities.
        batch_size (Optional[PositiveInt]): The batch size.
        backprop_every (Optional[PositiveInt]): When to backprop in training. If we have small batches due to memory limitation we may want to run backprop less than every iteration so that we in practice get larger batch size and a more stable training.
        loss_weight_type (Optional[LossWeightType]): The type of loss weight. Options include:
            - basic: standard `1/(count + 1)`
            - max: sets the maximum for to `count = min(count + 1, max(1000, total_with_label / 3))`
            - sqrt: sets the weight to `1/sqrt(count + 1)`

    Methods:
        __init__(self, **data: Any) -> None: Initializes the Scenario_Cfg instance and performs validation on scene names. Also checks and sets the batch size from global configurations if provided.
    """

    name: str = ...
    "The name of the scenario, provided separately"

    graph_name: str = ...
    "The name of the graph that we want to be using"

    optimizer_type: OptimizerType = ...
    "The type of optimizer to deploy"

    main_task: Optional[str] = Field(default=None)
    "The name of the task, if ignored it will default to the first task in the scene task list"

    jitter_pool: JitterPool = ...
    "How to pool jittered images"

    view_pool: str = ...
    "How to merge the views when outcomes are pooled on exam level"

    scenes: list[Scene_Cfg_Raw] = Field(default_factory=list, min_items=1)
    "The scenes that we want to run"

    min_channels: Optional[PositiveInt] = Field(default=None)
    "The minimum number of channels to use for the implicit modalities"

    batch_size: Optional[PositiveInt] = Field(default=None)
    "The batch size"

    backprop_every: Optional[PositiveInt] = Field(default=None)
    """When to backprop in training

    If we have small batches due to memory limitation we may want to run backprop less than every iteration
    so that we in practice get larger batch size and a more stable training."""

    loss_weight_type: Optional[LossWeightType] = Field(default=LossWeightType.basic)
    """The type of loss weight

    We want the loss to be multiplied by a factor based on the frequency. By default
    we have used the `1/(count + 1)` for generating the factor by which we want to multiple.
    This will cause rare classes to be learnable but may also hinder the network from
    learning anything about a label's mode. Thus we have the options:

    * basic - standard `1/(count + 1)`
    * max - sets the maximum for to `count = min(count + 1, max(1000, total_with_label / 3))`
    * sqrt - sets the weight to `1/sqrt(count + 1)`
    """

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        scene_names = [s.name for s in self.scenes]
        assert len(set(scene_names)) == len(
            scene_names
        ), f'Duplicated scene names: "{", ".join(scene_names)}"'

        from global_cfgs import Global_Cfgs
        from UIs.console_UI import Console_UI

        if user_batch_size := Global_Cfgs().get("batch_size"):
            if self.batch_size is not None:
                Console_UI().inform_user(
                    f"Using batch size {user_batch_size} instead of the scenario defined {self.batch_size}"
                )
            self.batch_size = user_batch_size
        elif self.batch_size is None:
            raise ValueError(
                "You must either provide batch size in argument -batch_size"
                + " or as a config option to the scenario"
            )
