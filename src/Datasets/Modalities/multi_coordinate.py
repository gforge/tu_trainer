from typing import List
import numpy as np

from DataTypes.Modality.csv import Modality_Csv_Column_Prefixes_Cfg

from .multi_regression import Multi_Regression
from Utils.ImageTools import SpatialTransform


class Multi_Coordinate(Multi_Regression[Modality_Csv_Column_Prefixes_Cfg]):

    def get_item(self, index, num_views=None):
        values = np.array(self.content.loc[index, :])
        values = values.reshape(num_views, 1, self.num_channels // 2, 2)
        values = np.repeat(values, self.num_jitters, axis=1)
        return {self.get_batch_name(): values}

    def post_get_batch(self, batch):
        """
        Hook for adding jitter rotation to coordinates
        """
        unrotated_target = batch[self.get_batch_name()]
        rotated_target = unrotated_target.copy()
        empty_rows = np.isnan(unrotated_target).any(axis=3)
        for i in range(unrotated_target.shape[0]):
            for j in range(unrotated_target.shape[1]):
                sp = SpatialTransform.from_pickled(batch['spatial_transforms'][i, j])
                if empty_rows[i, j].all():
                    continue

                # This rotation could probably be in the loading step
                active_rows = ~empty_rows[i, j]
                # Copy or we will end up messing up targets as the conversion is in place
                original_target = unrotated_target[i, j, active_rows, :].copy()
                rotated_target[i, j, active_rows, :] = sp.convert_original_points_to_processed_image_space(
                    original_points=original_target)

        batch[f'{self.get_batch_name()}_rotated'] = rotated_target
        batch[f'{self.get_batch_name()}_empty_rows'] = empty_rows

    def get_loss_type(self):
        return 'mse_with_spatial_transform'

    def _assert_modality_cfgs(self):
        assert isinstance(self._cfgs, Modality_Csv_Column_Prefixes_Cfg)

    @staticmethod
    def prefix_to_column_names(prefix: str):
        return [f'{prefix}_{suffix}' for suffix in ['x', 'y']]

    def get_column_names(self) -> List[str]:
        self._assert_modality_cfgs()

        columns = []
        for prefix in self._cfgs.column_prefixes:
            columns.extend(Multi_Coordinate.prefix_to_column_names(prefix=prefix))

        return columns
