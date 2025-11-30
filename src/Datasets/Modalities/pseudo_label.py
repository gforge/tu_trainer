from DataTypes import Any_CSV_Modality_Cfgs
from .Base_Modalities.base_number import Base_Number
from .Base_Modalities.base_output import Base_Output
from .Base_Modalities.base_distribution import Base_Distribution


class Pseudo_Label(Base_Number[Any_CSV_Modality_Cfgs], Base_Output, Base_Distribution):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def num_channels(self):
        if self._cfgs.num_channels is None:
            return 128
        return self._cfgs.num_channels

    def has_classification_loss(self):
        return False

    def get_default_model_cfgs(self):
        return {
            'model_type': 'One_to_One',
            'neural_net_cfgs': {
                'neural_net_type': 'Fully_Connected',
                'num_hidden': 0,
                'consistency': self.get_consistency(),
            }
        }

    def get_implicit_modality_cfgs(self):
        return {
            'type': 'Implicit',
            'num_channels': max(8, self.num_channels),
            'has_reconstruction_loss': False,
            'consistency': self.get_consistency(),
        }
