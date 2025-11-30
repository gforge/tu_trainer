from typing import Literal

from pydantic.types import PositiveInt

from .other import Loss_Csv_Cfg_Base


class Loss_Wasserstein_GAN_GP_Loss_Cfgs(Loss_Csv_Cfg_Base):
    loss_type: Literal['wGAN_gp'] = ...
    "The type of classification loss"

    update_model_every_step: PositiveInt = 5
