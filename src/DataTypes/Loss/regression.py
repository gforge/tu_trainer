from typing import List, Literal

from .other import Loss_Csv_Cfg_Base


class Loss_Regression_Cfgs(Loss_Csv_Cfg_Base):
    loss_type: Literal['mse_loss', 'mse_with_spatial_transform', 'mse_with_spatial_transform_and_line',
                       'mse_with_spatial_transform_and_independent_line']
    "Regression loss types"

    ignore_index: int = ...
    "The target index that is to be masked/ignored"

    to_each_view_its_own_label: bool = True
    "Should the loss be evaluated on a per view basis or should the interpretation be per exam?"

    output_shape: List[int] = ...

    regression: bool = True


class Loss_Regression_Line_Cfgs(Loss_Regression_Cfgs):
    skip_line_loss: bool = False


class Loss_Regression_Independent_Line_Cfgs(Loss_Regression_Line_Cfgs):
    skip_independent_line_loss: bool = False
