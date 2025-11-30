from typing import Generic, TypeVar
import torch
from torch import Tensor
from DataTypes import Loss_Regression_Cfgs, Loss_Regression_Line_Cfgs, Loss_Regression_Independent_Line_Cfgs

from GeneralHelpers import init_nan_array
from .csv_loss import CSV_Loss
from Utils.ImageTools import SpatialTransform

Regressors = TypeVar('Regressors', Loss_Regression_Cfgs, Loss_Regression_Line_Cfgs,
                     Loss_Regression_Independent_Line_Cfgs)


class MSE_WSP_Loss(Generic[Regressors], CSV_Loss[Regressors]):
    """Mean square errors with spatial transformation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pi_model = False
        self.classification = False
        self.reconstruction = False
        self.regression = True

    @property
    def results_output_name(self):
        return 'coordinates'

    def pool_and_reshape_output(self, output, num_views):
        # we want the values to be between 0 to 1
        return output

    def pool_and_reshape_target(self, target):
        return target

    def calculate_loss(self, output, target):
        return 0

    def _calculate_loss(self, target: Tensor, output: Tensor, empty_rows: Tensor):
        loss = 0
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                if empty_rows[i, j].all():
                    continue

                active_rows = ~empty_rows[i, j]
                diff = output[i, j, active_rows] - target[i, j, active_rows]
                loss += torch.sqrt((diff**2).sum())

        return {'loss': loss, 'results': {}}

    def calculate_regression_loss(self, output, target, batch):
        rotated_target = batch[f'{self.target_name}_rotated']
        rotated_target = rotated_target.to(device=output.device, dtype=torch.float32)
        empty_rows = batch[f'{self.target_name}_empty_rows']
        empty_rows = empty_rows.to(output.device)
        rotated = derotate_output_to_original_coordinates(
            spatial_transforms=batch['spatial_transforms'],
            output=output,
            target=target,
        )
        output = output.view(rotated_target.shape)

        loss_dict = self._calculate_loss(target=rotated_target, output=output, empty_rows=empty_rows)

        mod_id = self.modality.get_name()  # Just shorter and easier to read

        if mod_id not in batch['results']:
            batch['results'][mod_id] = {}

        batch['results'][mod_id].update({
            # This are the coordinate in the source image
            'output': rotated['original_coordinates'],
            'target': target,
            self.results_output_name: {
                'predicted': output.view(target.shape).detach().cpu().numpy(),
                'target': rotated_target,
                **loss_dict['results']
            }
        })

        return loss_dict['loss'] / target.shape[0] / target.shape[1]


def derotate_output_to_original_coordinates(
    spatial_transforms: Tensor,
    output: Tensor,
    target: Tensor,
):
    org_output_shape = output.shape
    # The positions _without_ the rotation + crop jitter
    original_coordinates = init_nan_array(target.shape, dtype=float)

    im_output = output.view(target.shape).detach().cpu().numpy().copy()
    for i in range(target.shape[0]):
        for j in range(target.shape[1]):
            sp = SpatialTransform.from_pickled(spatial_transforms[i, j])

            original_coordinates[i, j, :, :] = sp.convert_processed_image_points_to_original_space(im_output[i, j])

    return {
        # The original coordinates has to have the original shape or runtime_value will have issues
        'original_coordinates': original_coordinates.reshape(org_output_shape),
    }
