import math
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Generic

from .base_number import Base_Number, Modality_Type
from .base_csv import Base_CSV


class Base_Value(Generic[Modality_Type], Base_Number[Modality_Type], Base_CSV, metaclass=ABCMeta):

    @abstractmethod
    def get_loss_type(self):
        pass

    @abstractmethod
    def collect_statistics(self):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cfgs.num_channels = len(self.content.columns)

        self.label_stats = {}

        self.prep_content()
        self.collect_statistics()

    @property
    def ignore_index(self) -> int:
        return self._cfgs.ignore_index

    @property
    def to_each_view_its_own_label(self) -> bool:
        return self._cfgs.to_each_view_its_own_label

    def has_regression_loss(self):
        return True

    def prep_content(self):
        pass

    def has_pseudo_label(self):
        return False

    def get_label_dictionary(self):
        pass

    def get_regression_loss_cfgs(self) -> Dict[str, Any]:
        return {
            'loss_type': self.get_loss_type(),
            'modality_name': self.get_name(),
            'output_name': self.get_encoder_name(),
            'target_name': self.get_batch_name(),
            'ignore_index': self.ignore_index,
            'to_each_view_its_own_label': self.to_each_view_its_own_label,
            'output_shape': self.get_shape()
        }

    def get_default_model_cfgs(self) -> Dict[str, Any]:
        return {
            'model_type': 'One_to_One',
            'neural_net_cfgs': {
                'neural_net_type': 'Fully_Connected',
                'num_hidden': 1,
                'consistency': self.get_consistency(),
            }
        }

    def get_implicit_modality_cfgs(self) -> Dict[str, Any]:
        nc = max(
            self._min_channels,
            2**math.ceil(math.log2(self.get_num_classes())),
        )
        return {
            'type': 'Implicit',
            'num_channels': nc,
            'has_reconstruction_loss': False,
            'consistency': self.get_consistency(),
        }
