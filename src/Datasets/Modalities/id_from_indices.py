import numpy as np

from DataTypes import Modality_ID_Cfg
from GeneralHelpers.pytorch_wrapper import wrap, unwrap
from Graphs.Losses.helpers import normalized_euclidean_distance
from .Base_Modalities.base_number import Base_Number
from .Base_Modalities.base_output import Base_Output
from .Base_Modalities.base_implicit import Base_Implicit
from .Base_Modalities.base_runtime_value import Base_Runtime_Value


class ID_from_Indices(Base_Number[Modality_ID_Cfg], Base_Output, Base_Implicit, Base_Runtime_Value):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Content is a pd.Series with -100 defined in get_modality_and_content
        self.content = kwargs.pop('content')
        self.to_each_view_its_own_label = True

    def has_classification_loss(self):
        return False

    def has_identification_loss(self):
        return True

    def has_reconstruction_loss(self):
        return False

    def get_identification_loss_name(self):
        return '%s_triplet_metric' % self.get_name()

    def get_identification_loss_cfgs(self):
        return {
            'loss_type': 'triplet_metric',
            'modality_name': self.get_name(),
            'output_name': self.get_encoder_name(),
            'target_name': self.get_batch_name(),
            'output_shape': self.get_shape(),
            'to_each_view_its_own_label': True,
        }

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
            'num_channels': self.num_channels,
            'has_reconstruction_loss': False,
            'consistency': self.get_consistency(),
        }

    def get_item(self, index: int, num_views: int):
        return {self.get_batch_name(): np.array([index] * (num_views * self.num_jitters))}

    def analyze_modality_specific_results(self, batch):
        distance = unwrap(batch['results'][self.get_name()].pop('euclidean_distance'))
        target = unwrap(batch['results'][self.get_name()].pop('target'))

        accuracy = self.compute_accuracy(distance=distance, target=target)
        self.set_runtime_value(runtime_value_name='accuracy',
                               value=accuracy,
                               indices=batch['indices'],
                               num_views=batch['num_views'])

        batch['results'][self.get_name()].update({'accuracy': accuracy.mean()})

    def compute_accuracy(self, output=None, target=None, distance=None):
        """
        Compute the accuracy for the euclidean distance

        See if the closes n-1 images are correct

        The inputs are numpy tensors of dimension:
        - output.shape = [batch_size, 256]
        - target.shape = [batch_size, 1]
        """
        assert output is not None or distance is not None, 'Either output or distance is required'
        assert target is not None, 'Target is required despite the None in the parameters'
        if distance is None:
            distance = unwrap(normalized_euclidean_distance(wrap(output)))

        target = target.repeat(axis=1, repeats=target.shape[0])
        target = target == target.T
        tsum = target.sum(axis=1)

        tsort_idx = target.argsort(axis=1)[:, ::-1]  # Sorts descending
        dist_idx = distance.argsort(axis=1)

        matches = [(tsort_idx[i, :tsum[i]] == np.sort(dist_idx[i, :tsum[i]])[::-1]) for i in range(tsum.shape[0])]
        # We need to subtract self as this will always match and is trivial
        accuracy = [max(0, match.sum() - 1) / (len(match) - 1.0) if len(match) > 1 else np.nan for match in matches]
        if len(accuracy) == 0:
            return None

        return np.stack(accuracy)
