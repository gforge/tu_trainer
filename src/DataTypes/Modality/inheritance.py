from pydantic import Field
from typing import Optional
from DataTypes.enums import LossWeightType

from DataTypes.general import BaseModelWithGet


class Dataset_Modality_Cfg_Inheritance(BaseModelWithGet):
    """Variables that can be supplemented by configs higher up in the hierarchy
    """
    scene_name: Optional[str] = Field(default=None)
    "The name of the scene"

    task_name: Optional[str] = Field(default=None)
    "The name of the task"

    loss_weight_type: Optional[LossWeightType] = Field(default=None)
    """The type of loss weight

    We want the loss to be multiplied by a factor based on the frequency. By default
    we have used the `1/(count + 1)` for generating the factor by which we want to multiple.
    This will cause rare classes to be learnable but may also hinder the network from
    learning anything about a label's mode. Thus we have the options:

    * basic - standard `1/(count + 1)`
    * max - sets the maximum for to `count = min(count + 1, max(1000, total_with_label / 3))`
    * sqrt - sets the weight to `1/sqrt(count + 1)`
    """
