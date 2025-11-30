from typing import List, Literal
from pydantic.types import confloat

from DataTypes.enums import JitterPool

from DataTypes.Task.activators import Task_Cfg_Activators
from .other import Loss_Csv_Cfg_Base
from pydantic import Field
from typing_extensions import Annotated


class Loss_Classification_Cfgs(Loss_Csv_Cfg_Base):
    loss_type: Literal['bipolar_margin_loss', 'cross_entropy', 'hierarchical_bce'] = ...
    "The type of classification loss"

    output_shape: List[int] = ...

    apply: Task_Cfg_Activators = ...
    "The active losses"

    signal_to_noise_ratio: Annotated[float, Field(gt=0.5, lt=1)] = 0.9
    """The signal to noise ratio as defined in the dataset config, defaults to 0.9

    Must be 0.5 < x < 1.0
    """

    view_pool: str = "max"
    "How to pool teh views. Usually defined in scenario as it is the same for all. Defaults to max."

    jitter_pool: JitterPool = JitterPool.mean
    "How to poll the difrerent jitters. Defaults to mean."
