import math
from typing import Generic
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from DataTypes.Modality.cfg_groups import CSV_Type

from GeneralHelpers.pytorch_wrapper import wrap
from .base_number import Base_Number
from .base_csv import Base_CSV


class Base_Label(Generic[CSV_Type], Base_Number[CSV_Type], Base_CSV, metaclass=ABCMeta):

    @abstractmethod
    def get_loss_type(self):
        pass

    @abstractmethod
    def collect_statistics(self):
        pass

    def save_dictionary(self):
        pass  # Not always implemened

    @abstractmethod
    def get_num_classes(self):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_index = self._cfgs.ignore_index
        self.signal_to_noise_ratio = self._cfgs.signal_to_noise_ratio
        self.to_each_view_its_own_label = self._cfgs.to_each_view_its_own_label

        self.criterion_weight = None
        self.proprity_views = []

        self.label_to_url = {}
        self.label_to_desc = {}

        self.label_stats = {}

        self.dictionary = kwargs['dictionary']
        self.save_dictionary()

        self.prep_content()
        self.convert_class_names_to_indices()
        self.collect_statistics()

        no = self.get_num_classes()
        self.set_channels(no)

        self._wrapped_weights = None

    def has_classification_loss(self):
        return True

    def prep_content(self):
        self.content = self.content.apply(lambda x: x.lower() if (isinstance(x, str)) else x)

    def convert_class_names_to_indices(self):
        assert self.dictionary is not None, f'No dictionary has been generated for {self.get_name()}'
        if len(self.cls_name_to_label) < 2:
            raise ValueError(f'No valid conversion has been generated for {self.get_name()}: {self.cls_name_to_label}')

        if self.content.dtype in ['int']:
            self.labels = self.content.copy().fillna(self.ignore_index)
        else:
            self.labels = self.content.map(self.cls_name_to_label).fillna(self.ignore_index).astype(int)

        assert self.labels.dtype in ['int'], f'Unknown label type "{str(self.labels.dtype)}"'

        return self

    @property
    def cls_name_to_label(self):
        assert self.dictionary is not None, f'No dictionary initiated for {self.get_name()}'
        assert isinstance(self.dictionary, pd.DataFrame), 'No dictionary set while trying to retrieve class name'

        cls_name_to_label = {}
        for i in range(len(self.dictionary)):
            name = self.dictionary['name'].iloc[i]
            label = self.dictionary['label'].iloc[i]

            cls_name_to_label[str(name).lower()] = label

        return cls_name_to_label

    def has_pseudo_label(self):
        return True

    def get_label_dictionary(self):
        return self.cls_name_to_label

    def get_item(self, index, num_views=None):
        if self.to_each_view_its_own_label:
            labels = np.array(self.labels[index])
        else:
            # All labels are assumed to be identical, hence [0]
            labels = np.array(self.labels[index][0]).reshape(-1,)
        # TODO: check that labels is float32 | int

        return {self.get_batch_name(): labels}

    def get_classification_loss_cfgs(self):
        return {
            'loss_type': self.get_loss_type(),
            'modality_name': self.get_name(),
            'output_name': self.get_encoder_name(),
            'target_name': self.get_batch_name(),
            'loss_weight': self.get_loss_weight(),
            'ignore_index': self.ignore_index,
            'signal_to_noise_ratio': self.signal_to_noise_ratio,
            'to_each_view_its_own_label': self.to_each_view_its_own_label,
            'output_shape': self.get_shape()
        }

    def get_default_model_cfgs(self):
        return {
            'model_type': 'One_to_One',
            'neural_net_cfgs': {
                'neural_net_type': 'Fully_Connected',
                'num_hidden': 1,
                'consistency': self.get_consistency(),
            }
        }

    def get_implicit_modality_cfgs(self):
        nc = max(
            self._min_channels,
            self.gpu_optimized_num_channels(),
        )
        return {
            'type': 'Implicit',
            'num_channels': nc,
            'has_reconstruction_loss': False,
            'consistency': self.get_consistency(),
        }

    def get_loss_weight(self):
        if self._wrapped_weights is None:
            self._wrapped_weights = wrap(self.label_stats['loss_weight'])
        return self._wrapped_weights

    def get_mean_accuracy(self, unfiltered_accuracy):
        accuracy = unfiltered_accuracy[unfiltered_accuracy != self.ignore_index]
        if len(accuracy) == 0:
            return None
        return np.mean(accuracy)

    def is_classification(self):
        return True

    def is_regression(self):
        return False

    def gpu_optimized_num_channels(self) -> int:
        return 2**math.ceil(math.log2(self.get_num_classes()))
