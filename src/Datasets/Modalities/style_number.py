from DataTypes import Modality_Style_Cfg
from .Base_Modalities.base_style import Base_Style
from .Base_Modalities.base_number import Base_Number


class Style_Number(Base_Style, Base_Number[Modality_Style_Cfg]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_default_model_cfgs(self):
        return {
            'model_type': 'One_to_One',
            'neural_net_cfgs': {
                'neural_net_type': 'Fully_Connected',
                'num_hidden': 1,
                'consistency': self.get_consistency(),
            }
        }
