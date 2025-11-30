from typing import Generic, List, TypeVar
import numpy as np
from DataTypes import Modality_Csv_Column_Prefixes_Cfg
from DataTypes.Modality.csv import Modality_Csv_Mutliple_Columns_Cfg

from GeneralHelpers.pytorch_wrapper import unwrap
from .Base_Modalities.base_value import Base_Value
from .Base_Modalities.base_output import Base_Output
from .Base_Modalities.helpers.dim import fix_dims

RegressionType = TypeVar('RegressionType', Modality_Csv_Column_Prefixes_Cfg, Modality_Csv_Mutliple_Columns_Cfg)


class Multi_Regression(Generic[RegressionType], Base_Value[RegressionType], Base_Output):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cfgs.num_channels = len(self.content.columns)

    def get_num_classes(self):
        return self.num_channels

    def get_item(self, index, num_views=None):
        values = np.array(self.content.loc[index, :])
        values = values.reshape(num_views, 1, self.num_channels)
        values = np.repeat(values, self.num_jitters, axis=1)
        return {self.get_batch_name(): values}

    def compute_accuracy(self, output, target):
        # For some reason the dimension can get lost and we get only a single value
        # TODO: fix output dim lost
        output = fix_dims(output)
        target = fix_dims(target)
        output = output.reshape(target.shape)

        diff = output - target
        return np.nanmean(np.abs(diff), axis=0)

    def analyze_modality_specific_results(self, batch):
        output = unwrap(batch['results'][self.get_name()].pop('output'))
        target = unwrap(batch['results'][self.get_name()].pop('target'))

        results = {}
        for i, k in enumerate(self.content.columns):
            self.set_runtime_value(runtime_value_name='output',
                                   subgroup_name=k,
                                   value=output[:, i],
                                   indices=batch['indices'],
                                   num_views=batch['num_views'])

            results['accuracy'] = self.compute_accuracy(target, output).mean()

        batch['results'][self.get_name()].update(results)

    def report_modality_specific_epoch_summary(self, summary):
        pass  # Not implemented

    def report_runtime_value(self, runtime_value_name, value, indices, num_views):
        pass  # Not implemented

    def has_regression_loss(self):
        return True

    def has_pseudo_label(self):
        # TODO: pseudo labels should probably exist for continuous variables as well, the text may
        # contain information about angles that could help positioning
        return False

    def get_loss_type(self):
        return 'mse_loss'

    def collect_statistics(self):
        self.label_stats['std'] = self.content.values.std()
        self.label_stats['max'] = self.content.values.max()
        self.label_stats['min'] = self.content.values.min()

    def get_column_names(self) -> List[str]:
        assert isinstance(self._cfgs, Modality_Csv_Mutliple_Columns_Cfg)

        return self._cfgs.columns
