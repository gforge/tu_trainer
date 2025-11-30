from torch.nn.functional import mse_loss
from torch import Tensor
from .mse_line_loss import MSE_Line_Loss, Loss_Regression_Independent_Line_Cfgs


class MSE_Independent_Line_Loss(MSE_Line_Loss[Loss_Regression_Independent_Line_Cfgs]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rows_per_independent_line_unit = 4

    @property
    def _skip_independent_line_loss(self) -> bool:
        return self._cfgs.skip_independent_line_loss

    @property
    def results_output_name(self):
        return 'independent_lines'

    def _calculate_loss(self, target: Tensor, output: Tensor, empty_rows: Tensor):
        loss = super()._calculate_loss(target=target, output=output, empty_rows=empty_rows)
        if self._skip_independent_line_loss or self._skip_line_loss:
            return loss

        assert target.shape[2] % self._rows_per_independent_line_unit == 0, 'Unexpected number of coordinates'

        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                for grp_no in range(int(target.shape[2] / self._rows_per_independent_line_unit)):
                    grp_idx = grp_no * self._rows_per_independent_line_unit
                    grp_end_idx = grp_idx + self._rows_per_independent_line_unit
                    if empty_rows[i, j, grp_idx:grp_end_idx].any():
                        continue

                    target_angle_relation = self._get_angle_relation(
                        data=loss['angle']['target'][i, j],
                        grp_idx=grp_idx,
                    )
                    output_angle_relation = self._get_angle_relation(
                        data=loss['angle']['output'][i, j],
                        grp_idx=grp_idx,
                    )

                    loss['loss'] += mse_loss(input=output_angle_relation, target=target_angle_relation)

        return loss

    def _get_angle_relation(self, data: Tensor, grp_idx: int):
        line_idx_1 = int(grp_idx / self._rows_per_line_unit)
        line_idx_2 = line_idx_1 + 1
        angles_1 = data[line_idx_1]
        angles_2 = data[line_idx_2]

        return angles_1 - angles_2
