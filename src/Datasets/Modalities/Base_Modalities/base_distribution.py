import numpy as np

from abc import ABCMeta

from DataTypes.Modality.style import Modality_Style_Cfg

from .base_explicit import Base_Explicit


class Base_Distribution(Base_Explicit[Modality_Style_Cfg], metaclass=ABCMeta):

    def get_item(self, index, num_views=None):
        return {self.get_batch_name(): self.generate_random_sample()}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.distribution == 'gaussian':
            self.generate_random_sample = self.generate_gaussian_sample

    @property
    def mean(self):
        return self._cfgs.mean

    @property
    def std(self):
        return self._cfgs.std

    @property
    def distribution(self):
        return self._cfgs.distribution

    def generate_gaussian_sample(self):
        gaussian = self.std * np.random.randn(*self.get_shape()) + self.mean
        return gaussian.astype('float32')

    # TODO add more distributions

    def is_distribution(self):
        return True
