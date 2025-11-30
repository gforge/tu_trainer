from typing import Any
from pydantic.types import confloat
import torch
import numpy as np
from pydantic import Field, ConfigDict, fields

from GeneralHelpers.pytorch_wrapper import wrap

from DataTypes.enums import OptimizerType
from DataTypes.general import BaseModelWithGet
from DataTypes.Task.activators import Task_Cfg_Activators
from typing_extensions import Annotated


class Loss_Cfg_Base(BaseModelWithGet):
    node_type: str = 'loss'
    "Just det definition of the node class"

    modality_name: str = ...
    "The name of the modality"

    output_name: str = ...
    "The name of the output"

    target_name: str = ...
    "The name of the target that is expected by the loss"

    apply: Task_Cfg_Activators = ...
    "The active losses"

    optimizer_type: OptimizerType = ...
    "The optimizer to use"

    has_pseudo_labels: bool = ...
    "Whether there are activated pseudo labels"

    pseudo_loss_factor: float = ...
    "How much pseudo loss from fictive labels compares to regular loss from manually annotated images"

    initial_learning_rate: float = ...
    "The initial learning rate at the start of the epoch"

    min_channels: Annotated[float, Field(ge=2)] = ...
    'Minimum representation size for when classes are smaller than a specific dim'


class Loss_Csv_Cfg_Base(Loss_Cfg_Base):
    loss_weight: torch.Tensor = fields.Field(default=None)
    "The loss weights associated with the outcomes"

    ignore_index: int = ...
    "The target index that is to be masked/ignored"

    to_each_view_its_own_label: bool = ...
    "Should the loss be evaluated on a per view basis or should the interpretation be per exam?"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        if isinstance(self.loss_weight, np.ndarray):
            self.loss_weight = wrap(self.loss_weight)
