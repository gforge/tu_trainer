from abc import ABCMeta
from typing import Generic

from DataTypes import Modality_Type
from .base_explicit import Base_Explicit


class Base_Output(Generic[Modality_Type], Base_Explicit[Modality_Type], metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def is_output_modality(self):
        return True

    def get_batch_name(self):
        return f'target_{self.get_name()}'

    # Output distributions do not decode
    def get_decoder_name(self):
        return self.get_encoder_name()

    def get_classification_loss_name(self):
        return f'{self.get_name()}_cls_loss'

    def get_classification_loss_cfgs(self):
        return None

    def get_regression_loss_name(self):
        return f'{self.get_name()}_reg_loss'

    def get_regression_loss_cfgs(self):
        return None

    def has_pseudo_label(self):
        return False
