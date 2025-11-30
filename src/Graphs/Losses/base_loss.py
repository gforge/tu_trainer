from __future__ import annotations
from typing import Generic, Optional, TYPE_CHECKING
import torch.nn as nn
from abc import ABCMeta, abstractmethod

from global_cfgs import Global_Cfgs
from DataTypes import Loss_Cfg_Type

if TYPE_CHECKING:
    from Datasets.experiment_set import Experiment_Set
    from Graphs.Models.model import Model


class Base_Loss(Generic[Loss_Cfg_Type], nn.Module, metaclass=ABCMeta):

    def __init__(
        self,
        experiment_set: Experiment_Set,
        loss_name: str,
        loss_cfgs: Loss_Cfg_Type,
    ):

        super().__init__()
        self.experiment_set = experiment_set
        self._loss_name = loss_name
        self._cfgs = loss_cfgs
        self.modality = self.experiment_set.get_modality(self.modality_name)
        self.tensor_shape = self.modality.get_tensor_shape()
        self.coef = 1.  # Default loss factor for all losses

        self.neural_net: Optional[Model] = None
        self.use_cuda = Global_Cfgs().get('DEVICE_BACKEND').lower() == 'cuda'

    @property
    def modality_name(self) -> str:
        return self._cfgs.modality_name

    @property
    def initial_learning_rate(self) -> float:
        return self._cfgs.initial_learning_rate

    @property
    def output_name(self) -> str:
        return self._cfgs.output_name

    @property
    def target_name(self) -> str:
        return self._cfgs.target_name

    @abstractmethod
    def forward(self, batch):
        pass

    def pool_and_reshape_output(self, output, num_views=None):
        output = output.view([-1, *self.tensor_shape])
        return output

    def pool_and_reshape_target(self, target, num_views=None):
        target = target.view([-1, *self.tensor_shape])
        return target

    def get_name(self):
        return self._loss_name

    def zero_grad(self):
        if self.neural_net is not None:
            self.neural_net.zero_grad()

    def step(self):
        if self.neural_net is not None:
            self.neural_net.step()

    def train(self):
        if self.neural_net is not None:
            self.neural_net.train()

    def eval(self):
        if self.neural_net is not None:
            self.neural_net.eval()

    def save(self, scene_name):
        if self.neural_net is not None:
            self.neural_net.save(scene_name)

    def update_learning_rate(self, learning_rate):
        if self.neural_net is not None:
            self.neural_net.update_learning_rate(learning_rate)
        else:
            self._cfgs.initial_learning_rate = learning_rate

    def update_stochastic_weighted_average_parameters(self):
        if self.neural_net is not None:
            self.neural_net.update_stochastic_weighted_average_parameters()

    def prepare_for_batchnorm_update(self):
        if self.neural_net is not None:
            self.neural_net.prepare_for_batchnorm_update()

    def update_batchnorm(self, batch):
        if self.neural_net is not None:
            self.neural_net.update_batchnorm(batch)

    def finish_batchnorm_update(self):
        if self.neural_net is not None:
            self.neural_net.finish_batchnorm_update()
