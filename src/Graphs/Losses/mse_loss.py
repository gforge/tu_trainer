from torch.nn.functional import mse_loss

from DataTypes import Loss_Regression_Cfgs

from .csv_loss import CSV_Loss


class MSE_Loss(CSV_Loss[Loss_Regression_Cfgs]):
    """
    This class is not yet used or tested
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pi_model = False
        self.classification = False
        self.reconstruction = False
        self.regression = self._cfgs.apply.regression

    def pool_and_reshape_output(self, output, num_views):
        # we want the values to be between 0 to 1
        return output

    def pool_and_reshape_target(self, target):
        return target

    def calculate_loss(self, output, target):
        return 0

    def calculate_regression_loss(self, output, target, batch):
        loss = 0
        nan_mask = target.eq(target)
        if not nan_mask.any().item():
            return loss

        output = output.view(target.shape)
        target[nan_mask].to(output.dtype).dtype
        loss = mse_loss(input=output[nan_mask], target=target[nan_mask].to(output.dtype), reduction="sum")

        modId = self.modality.get_name()  # Just shorter and easier to read

        if modId not in batch['results']:
            batch['results'][modId] = {
                'output': output,
                'target': target,
            }

        return loss / target.shape[0] / target.shape[1]
