from typing import List
from .multi_line import Multi_Line


class Multi_Independent_Line(Multi_Line):

    def get_loss_type(self):
        return 'mse_with_spatial_transform_and_independent_line'

    @staticmethod
    def prefix_to_column_names(prefix: str):
        columns = []
        for line_no in ['line1', 'line2']:
            columns.extend(Multi_Line.prefix_to_column_names(prefix=f'{prefix}_{line_no}'))
        return columns

    def get_column_names(self) -> List[str]:
        self._assert_modality_cfgs()

        columns = []
        for prefix in self._cfgs.column_prefixes:
            columns.extend(Multi_Independent_Line.prefix_to_column_names(prefix=prefix))

        return columns
