import torch

from DataTypes import Loss_Triple_Metric_Cfgs

from .helpers import normalized_euclidean_distance
from .base_convergent_loss import Base_Convergent_Loss


class Triplet_Metric_Loss(Base_Convergent_Loss[Loss_Triple_Metric_Cfgs]):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.distance = None

    @property
    def margin(self):
        return self._cfgs.margin

    @property
    def output_shape(self):
        return self._cfgs.output_shape

    def _forward_analysis(self, batch, loss):
        if loss > 0:
            loss = loss.item()

        converted_distance = self.distance.detach().cpu().clone().numpy()
        self.distance = None

        batch['results'][self.modality.get_name()].update({'euclidean_distance': converted_distance})
        self.modality.analyze_results(batch, loss)

    def calculate_loss(self, output, target):
        """
        The inputs are torch tensors of dimension:
        - output.shape = [batch_size, 256]
        - target.shape = [batch_size, 1]
        """
        euclidian_dist = self.euclidean_norm_dist(output)
        target = target.expand([-1, target.shape[0]])
        target = target == target.t()

        # The batch consists of a single example
        if not any(target[0, :]) or not any(~target[0, :]):
            return 0

        batch_size = euclidian_dist.size(0)

        # The distance for the same patient should be as close as possible and this high values should give penalty
        furthest_pos_distances = torch.stack([torch.max(euclidian_dist[i, target[i, :]]) for i in range(batch_size)])
        # The non-matching should be as far away as possible, i.e. the smaller the distance the worse
        closest_neg_distances = torch.stack([torch.min(euclidian_dist[i, ~target[i, :]]) for i in range(batch_size)])

        # furthest positive distance should be at least, margin size smaller than the closest negative distance
        diff = (self.margin + furthest_pos_distances - closest_neg_distances).clamp(min=0)
        loss = diff.mean()
        return loss

    def pool_and_reshape_target(self, target, num_views=None):
        target = target.view(-1, 1)
        return target

    def pool_and_reshape_output(self, output, num_views=None):
        return output

    def euclidean_norm_dist(self, output):
        self.distance = normalized_euclidean_distance(output)
        return self.distance
