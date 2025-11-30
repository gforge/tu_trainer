import numbers
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from torch import Tensor
from DataTypes.Modality.cfg_groups import Any_CSV_Modality_Cfgs

from Datasets.Modalities.Base_Modalities.base_modality import Base_Modality

from .helpers.dim import fix_dims

_default_id = '@default@'


class Base_Runtime_Value(Base_Modality[Any_CSV_Modality_Cfgs]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.runtime_values = {}

    def get_default_value(self, runtime_value_name):
        if runtime_value_name in [
                'entropy',
                'accuracy',
                'output',
                'pseudo_entropy',
                'pseudo_accuracy',
                'pseudo_output',
        ]:
            return self._cfgs.ignore_index

        # TODO - prediction is only used in label and one vs rest but this should possibly be a number or NaN otherwise
        if runtime_value_name in ['prediction', 'pseudo_prediction']:
            return ''

        raise KeyError(f'Unknown runtime {runtime_value_name} for {self.get_name()}')

    def __get_runtime_name(self, runtime_value_name, column=None):
        runtime_value_name = runtime_value_name.lower()
        if column is None or column == _default_id:
            return f'{self.get_name()}_{runtime_value_name}'

        return f'{self.get_name()}_{column}_{runtime_value_name}'

    def get_initial_runtime_value(self, runtime_value_name: str):
        rv_name = runtime_value_name.lower()
        assert rv_name not in self.runtime_values,\
            f'Trying to init {rv_name} but it is already initialized in {self.get_name()}'

        runtime_value = self.content.copy()

        if isinstance(runtime_value, pd.Series):
            runtime_value = pd.DataFrame(data={_default_id: runtime_value})

        for column in runtime_value.columns:
            runtime_value[column] = self.__fill_runtime_value(runtime_value=runtime_value[column],
                                                              runtime_value_name=rv_name,
                                                              column=column)

        runtime_value = self.__rename_runtime_columns(runtime_value=runtime_value, runtime_value_name=rv_name)

        return runtime_value

    def __rename_runtime_columns(self, runtime_value: pd.DataFrame, runtime_value_name: str) -> pd.DataFrame:
        new_column_names = {n: n for n in runtime_value.columns}
        for old_column, column in new_column_names.items():
            new_column_names[old_column] = self.__get_runtime_name(runtime_value_name=runtime_value_name, column=column)
        runtime_value.rename(columns=new_column_names, inplace=True)
        return runtime_value

    def __fill_runtime_value(self, runtime_value: pd.Series, runtime_value_name: str, column: str = None):
        default_value = self.get_default_value(runtime_value_name=runtime_value_name)
        if isinstance(default_value, (numbers.Number, bool, str)):
            runtime_value.values.fill(default_value)
        elif isinstance(default_value, dict):
            runtime_value = runtime_value.map(default_value)
        return runtime_value

    def get_runtime_value(self, runtime_value_name: str, subgroup_name=None, convert_to_numeric=False):
        """
        runtime values are the values that are computed during
        the runtime, like entropy, accuracy, etc...
        We store them during training and testing to be able to
        measure performance.
        """
        rv_name = runtime_value_name.lower()

        if rv_name not in self.runtime_values:
            self.runtime_values[rv_name] = self.get_initial_runtime_value(runtime_value_name=rv_name)

        column = self.__get_runtime_name(runtime_value_name=rv_name, column=subgroup_name)
        if column not in self.runtime_values[rv_name]:
            raise KeyError(f'The {self.get_name()} failed to init runtime_values {rv_name} for {subgroup_name}.' +
                           f' Looking for key {column} but we have only initiated the keys: ' +
                           str(self.runtime_values[rv_name].keys()))

        rv = self.runtime_values[rv_name][column]

        if convert_to_numeric and is_numeric_dtype(rv):
            # The rv will be filled with NaN - to remove we could set .fillna(0, downcast='infer')
            # but this could cause issues as we don't really know what the runtime will be used to store
            rv = pd.to_numeric(rv, errors='coerce')

        return rv

    def has_runtime_value(self, name: str) -> bool:
        return name in self.runtime_values

    def get_runtime_values(self):
        values = []
        for rt_dataframe in self.runtime_values.values():
            [values.append(rt_dataframe[c]) for c in rt_dataframe]

        return values

    def set_runtime_value(self, runtime_value_name, value, indices, num_views=None, subgroup_name=None):
        runtime_value = self.get_runtime_value(runtime_value_name, subgroup_name=subgroup_name)
        value = fix_dims(value)

        if isinstance(indices, Tensor):
            indices = indices.detach().cpu().numpy()

        if self.to_each_view_its_own_label:
            assert num_views is not None, 'The num_views is required when values are specific to each image'
            c = np.cumsum([0, *num_views])
            for (i, idx) in enumerate(indices):
                runtime_value.at[idx] = value[c[i]:c[i + 1]]
        else:
            for (i, idx) in enumerate(indices):
                runtime_value.at[idx] = value[i]
