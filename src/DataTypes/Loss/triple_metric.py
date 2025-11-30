from typing import List, Literal

from .other import Loss_Csv_Cfg_Base


class Loss_Triple_Metric_Cfgs(Loss_Csv_Cfg_Base):
    loss_type: Literal['triplet_metric'] = ...
    "The type of classification loss"

    output_shape: List[int] = ...

    margin: float = 1.
    "The furthest distance that we penalize for"
