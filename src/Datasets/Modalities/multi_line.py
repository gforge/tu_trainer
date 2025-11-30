from typing import List
from .multi_coordinate import Multi_Coordinate


class Multi_Line(Multi_Coordinate):

    def get_loss_type(self):
        return 'mse_with_spatial_transform_and_line'

    @staticmethod
    def prefix_to_column_names(prefix: str):
        columns = []
        for point_name in ['c1', 'c2']:
            columns.extend(Multi_Coordinate.prefix_to_column_names(prefix=f'{prefix}_{point_name}'))
        return columns

    def get_column_names(self) -> List[str]:
        self._assert_modality_cfgs()

        columns = []
        for prefix in self._cfgs.column_prefixes:
            columns.extend(Multi_Line.prefix_to_column_names(prefix=prefix))

        return columns
