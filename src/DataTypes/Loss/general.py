from typing import TypeVar, Union
from pydantic import RootModel
from .classification import Loss_Classification_Cfgs
from .reconstruction import Loss_L2_Cfgs, Loss_L1_Laplacian_Pyramid_Cfgs
from .regression import Loss_Regression_Cfgs, Loss_Regression_Line_Cfgs, Loss_Regression_Independent_Line_Cfgs
from .triple_metric import Loss_Triple_Metric_Cfgs
from .wasserstain_gan import Loss_Wasserstein_GAN_GP_Loss_Cfgs

Any_Loss_Cfg = Union[
    Loss_Classification_Cfgs,
    Loss_L2_Cfgs,
    Loss_L1_Laplacian_Pyramid_Cfgs,
    Loss_Regression_Cfgs,
    Loss_Regression_Line_Cfgs,
    Loss_Regression_Independent_Line_Cfgs,
    Loss_Triple_Metric_Cfgs,
    Loss_Wasserstein_GAN_GP_Loss_Cfgs
]
Loss_Cfg_Type = TypeVar('Loss_Cfg_Type', bound=Any_Loss_Cfg)

CSV_Loss_Cfg = Union[
    Loss_Classification_Cfgs,
    Loss_Regression_Cfgs,
    Loss_Regression_Line_Cfgs,
    Loss_Regression_Independent_Line_Cfgs
]
CSV_Loss_Type = TypeVar('CSV_Loss_Type', bound=CSV_Loss_Cfg)


class Loss_Cfg_Parser(RootModel[Any_Loss_Cfg]):
    """
    Loss configuration parser
    """
    pass
