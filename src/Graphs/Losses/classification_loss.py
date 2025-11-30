from DataTypes import Loss_Classification_Cfgs

from .csv_loss import CSV_Loss
from .helpers import get_pool_network


class Classification_Loss(CSV_Loss[Loss_Classification_Cfgs]):
    "The core classification loss superclass"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regression = False
        self.__setup_jitter_pool()
        self.__setup_view_pool()

    def __setup_view_pool(self):
        if self.to_each_view_its_own_label:
            self.view_pool_net = None
            return

        self.view_pool_net = get_pool_network(
            view_pool=self._cfgs.view_pool,
            dim=1,
            modality_name=self.get_name(),
            keepdim=True,
        )

    def __setup_jitter_pool(self):
        self.jitter_pool_net = get_pool_network(
            view_pool=self._cfgs.jitter_pool.value,
            dim=0,
            modality_name=self.get_name(),
        )

    def pool_and_reshape_output(self, output, num_views):
        output = output.view([-1, *self.output_shape])
        output = output.permute([1, 0, 2])
        # At this point, the output is in the shape of:
        # [num_jitter x batch_size x num_classes]

        if self.view_pool_net is not None:
            output = self.view_pool_net(output, num_views=num_views)

        pi_output = output.clone()

        if self.jitter_pool_net is not None:
            output = self.jitter_pool_net(output)
        # Alight, now it's [batch_size x num_classes]

        return output, pi_output

    def pool_and_reshape_target(self, target):
        # TODO: Ali: This seems like a bad idea - it basically drops all the structure...
        return target.view(-1)
