import pandas as pd
import numpy as np
import torch
from torch import Tensor
from collections import defaultdict
from abc import ABCMeta
from typing import List, Union

from GeneralHelpers.pytorch_wrapper import unwrap
from .base_explicit import Base_Explicit
from .base_runtime_value import Base_Runtime_Value


class Base_CSV(Base_Explicit, Base_Runtime_Value, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_view = None

        assert('content' in kwargs and
               not kwargs['content'] is None and
               (isinstance(kwargs['content'], pd.Series) or
                isinstance(kwargs['content'], pd.DataFrame))),\
            'A CSV modality should either have a pd.Series, pd.DataFrame or None as content'

        self.dataset_name = kwargs.pop('dataset_name')
        self.content = kwargs.pop('content')

        self.__init_report_data()

    def is_csv(self):
        return True

    def get_sub_content(self, index, sub_index):
        content = 'not found!'
        try:
            content = self.content[index][sub_index]
        except KeyError:
            raise KeyError("Could not locate [{index}][{sub_index}] in the '{dataset}' dataset".format(
                index=index,
                sub_index=sub_index,
                dataset=self.dataset_name,
            ))
        return content

    def get_content(self, index):
        return [self.get_sub_content(index, i) for i in range(len(self.content[index]))]

    def _add_2_report_data(self, key: str, value: Union[Tensor, List]):
        if isinstance(value, list):
            assert isinstance(value[0], (int, np.integer)), \
                f'Cannot handle list/array for {key} as it is of int type: {type(value[0])}'
            value = torch.IntTensor(value)

        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)

        if not isinstance(value, Tensor):
            raise ValueError(f'Unexpected value for {key} type = {type(value)}')

        # Eject from GPU and clone - we don't want this to stick in the memory
        value = value.detach().cpu().clone()

        if key in self._data_2_report:
            self._data_2_report[key] = torch.cat((self._data_2_report[key], value), axis=0)
        else:
            self._data_2_report[key] = value

    def _get_items_from_report_data(self, items: List[str], clear_storage: bool) -> List[np.ndarray]:
        values = []
        for key in items:
            v = self._data_2_report[key]
            if isinstance(v, Tensor):
                v = unwrap(v)
            values.append(v)

        if clear_storage:
            self.__init_report_data()
        return values

    def __init_report_data(self):
        self._data_2_report = defaultdict(lambda: None)

    # Not used prior to name change - left in case we need it :-)
    # def previous_set_runtime_value(
    #         self,
    #         runtime_value_name,
    #         runtime_value_series,
    # ):
    #     assert isinstance(runtime_value_series, pd.Series), \
    #         f'In {self.get_name()}, {runtime_value_name} should be a pandas series.' + \
    #         f' Instead, found {str(type(runtime_value_series))}'

    #     name = runtime_value_name.lower()
    #     self.runtime_values[name] = runtime_value_series
    #     if (isinstance(runtime_value_series[0], str)):
    #         self.runtime_values[name] = self.runtime_values[name].apply(ast.literal_eval)
