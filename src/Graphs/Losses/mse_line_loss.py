from typing import Generic, TypeVar
import torch
from torch.nn.functional import mse_loss
from torch import Tensor
from GeneralHelpers import init_nan_array, init_torch_nan_array

from .mse_wsp_loss import MSE_WSP_Loss, Loss_Regression_Line_Cfgs, Loss_Regression_Independent_Line_Cfgs

LineRegressors = TypeVar('LineRegressors', Loss_Regression_Line_Cfgs, Loss_Regression_Independent_Line_Cfgs)


class MSE_Line_Loss(Generic[LineRegressors], MSE_WSP_Loss[LineRegressors]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rows_per_line_unit = 2

    @property
    def _skip_line_loss(self) -> bool:
        return self._cfgs.skip_line_loss

    @property
    def results_output_name(self):
        return 'lines'

    def _calculate_loss(self, target: Tensor, output: Tensor, empty_rows: Tensor):
        loss = super()._calculate_loss(target=target, output=output, empty_rows=empty_rows)
        if self._skip_line_loss:
            return loss

        assert target.shape[2] % self._rows_per_line_unit == 0, 'Unexpected number of coordinates'

        target_linespecifics = init_nan_array((*target.shape[0:2], 2))
        output_linespecifics = init_nan_array((*target.shape[0:2], 2))
        no_lines = int(target.shape[2] / self._rows_per_line_unit)
        shape = (*target.shape[:2], no_lines)
        target_angle = init_torch_nan_array(shape=shape, template_tensor=output)
        output_angle = init_torch_nan_array(shape=shape, template_tensor=output)

        target_distance = init_torch_nan_array(shape=shape, template_tensor=output)
        output_distance = init_torch_nan_array(shape=shape, template_tensor=output)

        loss['angle'] = {'target': target_angle, 'output': output_angle}
        loss['distance'] = {'target': target_distance, 'output': output_distance}

        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                for grp_no in range(no_lines):
                    idx = grp_no * self._rows_per_line_unit
                    end_idx = idx + 2
                    if empty_rows[i, j, idx:end_idx].any():
                        continue

                    line_target = target[i, j, idx:end_idx, :]
                    linespecific_targets = get_distance_and_angle(two_points_in_2x2_tensor=line_target)

                    line_output = output[i, j, idx:end_idx, :]
                    linespecific_output = get_distance_and_angle(two_points_in_2x2_tensor=line_output)
                    loss['loss'] += mse_loss(input=linespecific_output, target=linespecific_targets)

                    loss['distance']['target'][i, j, grp_no] = linespecific_targets[0]
                    loss['distance']['output'][i, j, grp_no] = linespecific_output[0]
                    loss['angle']['target'][i, j, grp_no] = linespecific_targets[1]
                    loss['angle']['output'][i, j, grp_no] = linespecific_output[1]

                    target_linespecifics[i, j] = linespecific_targets.detach().cpu().numpy()
                    output_linespecifics[i, j] = linespecific_output.detach().cpu().numpy()

        loss['results']['linespecifics'] = {'target': target_linespecifics, 'output': output_linespecifics}

        return loss


def get_distance_and_angle(two_points_in_2x2_tensor):
    c1 = two_points_in_2x2_tensor[0, :]
    c2 = two_points_in_2x2_tensor[1, :]

    diff = c1 - c2

    dist = torch.norm(diff, keepdim=True).view(-1)
    # Just a fallback for division by 0 when
    if dist == 0:
        angle = dist
    else:
        angle = torch.sin(diff[0] / dist).view(-1)

    return torch.cat((dist, angle))
