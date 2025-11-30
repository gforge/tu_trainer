from DataTypes import Modality_Implicit_Cfg
from .base_modality import Base_Modality


class Base_Implicit(Base_Modality[Modality_Implicit_Cfg]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def has_reconstruction_loss(self):
        return self._cfgs.has_reconstruction_loss

    def is_implicit_modality(self):
        return True

    def get_reconstruction_loss_name(self):
        return '%s_l2_reconst' % self.get_name()

    def get_reconstruction_loss_cfgs(self):
        return {
            'loss_type': 'l2_loss',
            'modality_name': self.get_name(),
            'relu': True,
            'tensor_shape': self.get_tensor_shape()
        }
