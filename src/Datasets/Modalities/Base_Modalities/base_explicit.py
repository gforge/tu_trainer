from typing import Any, Dict, Generic
import numpy as np
from abc import ABCMeta, abstractmethod
import time

from pydantic.main import BaseModel
from .base_modality import Base_Modality, Modality_Type


class Base_Explicit(Generic[Modality_Type], Base_Modality[Modality_Type], metaclass=ABCMeta):
    """
    Every explicit modality should have three functions:
    1- get_item() samples from the modality
    2- get_implicit_modality_cfgs() returns the implicit modality that
        the explicit modality prefers
    3- get_default_model_cfgs() returns a modal config that maps explicit modality
        to it's implicit modality
    4- is_input_modality() and is_output_modality()
        Explicit modalities are either input modalities or output modalities
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # if loading is process-heavy or disk-heavy,
        # this number will be updated in the inherited modality

        self.implicit_modality = None

    @abstractmethod
    def get_item(self, index: int, num_views=None):
        pass

    @abstractmethod
    def get_default_model_cfgs(self):
        pass

    @abstractmethod
    def get_implicit_modality_cfgs(self):
        pass

    def get_implicit_modality_name(self):
        return 'implicit_%s' % (self.get_name())

    def get_model_name(self):
        return '%s_path' % (self.get_name())

    def __get_model_cfgs_base_dict(self) -> Dict[str, Any]:
        model_cfgs = self._cfgs.model_cfgs
        if model_cfgs is None:
            return self.get_default_model_cfgs()

        if isinstance(model_cfgs, BaseModel):
            return model_cfgs.dict()

        raise ValueError('Unexpected model config')

    def get_model_cfgs(self) -> Dict[str, Any]:
        model_cfgs = self.__get_model_cfgs_base_dict()

        if self.is_input_modality():
            model_cfgs['heads'] = [self.get_name()]
            model_cfgs['tails'] = [self.get_implicit_modality_name()]
        elif self.is_output_modality():
            model_cfgs['tails'] = [self.get_name()]
            model_cfgs['heads'] = [self.get_implicit_modality_name()]

        return {**model_cfgs}

    def get_batch(self, batch):
        start_time = time.time()
        assert len(batch['indices']) == len(batch['num_views'])

        loaded_data = {}
        for index, num_views in zip(batch['indices'], batch['num_views']):
            # indices is a one dimensional vector of numbers with batch_size elements
            # sub_indices is a two dimensional vector or numbers with batch_size x num_views_per_sample
            # transforms is a tensor of batch_size x num_views_per_sample x num_jitters x 3 x 3
            # the output is going to be batch_size x modality_size
            single_item = self.get_item(index=index, num_views=num_views)
            for key, value in single_item.items():
                if key not in loaded_data:
                    loaded_data[key] = value
                else:
                    if type(value) is np.ndarray:
                        loaded_data[key] = np.concatenate([loaded_data[key], value])
                    elif key == 'spatial_transforms':
                        loaded_data[key].extend(value)
                    else:
                        raise KeyError(f'The key {key} is neither a numpy array or spatial transform')

        batch.update(loaded_data)
        batch['post_batch_hooks'].add(self.post_get_batch)
        batch['time']['load'][self.get_name()] = {'start': start_time, 'end': time.time()}
        return batch

    def post_get_batch(self, batch):
        """
        Hook for allowing post-retrieval modification to batch data
        e.g. adding jitter rotation to coordinates
        """
        return batch

    def is_explicit_modality(self):
        return True

    def is_csv(self):
        return False

    def is_distribution(self):
        return False
