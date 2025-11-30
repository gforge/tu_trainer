import torch.nn.functional as F

from DataTypes import Loss_L2_Cfgs

from .base_convergent_loss import Base_Convergent_Loss


class L2_Loss(Base_Convergent_Loss[Loss_L2_Cfgs]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_name = self.modality.get_decoder_name()
        self.target_name = self.modality.get_encoder_name()

    @property
    def relu(self) -> bool:
        return self._cfgs.relu

    def calculate_loss(self, output, target):
        if self.relu:
            return F.mse_loss(F.relu(output), F.relu(target).detach())

        return F.mse_loss(output, target.detach())
