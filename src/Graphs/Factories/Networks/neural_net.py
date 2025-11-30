import torch
from torch import nn
import torch.optim as optim
import numpy as np
from typing import Iterator
from DataTypes import Network_Cfg_Base
import time

from global_cfgs import Global_Cfgs
from .helpers import summarize_model_size
from file_manager import File_Manager
from UIs.console_UI import Console_UI
from DataTypes import OptimizerType


class Neural_Net(nn.Module):
    """Basic neural network class."""

    def __init__(
        self,
        neural_net_cfgs: Network_Cfg_Base,
        layers: nn.Module,
        optimizer_type: str,
        input_name: str = "",
        output_name: str = "",
        input_shape: list = [],
        output_shape: list = [],
        load_from_batch: bool = True,
    ):
        super().__init__()
        self.__neural_net_cfgs = neural_net_cfgs
        self.__input_name = input_name
        self.__output_name = output_name
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        self.layers = layers
        self.__optimizer_type = optimizer_type
        self.__load_from_batch = load_from_batch
        self.__weighted_average_parameters = None
        self.__weighted_average_parameters_counter = 0
        self.__batch_norm_update_counter = 0
        self.__momenta = {}

        if self.__load_from_batch:
            self.forward = self.forward_from_batch
        else:
            self.forward = self.forward_data

        if Global_Cfgs().get("DEVICE_BACKEND") == "cuda":
            self.layers.cuda()
            self.layers = nn.DataParallel(layers)

        self.network_memory_usage = None
        if Global_Cfgs().get("check_model_size"):
            try:
                self.network_memory_usage = summarize_model_size(
                    model=layers,
                    input_size=(*self.__input_shape,),
                    device=Global_Cfgs().get("DEVICE_BACKEND"),
                )
            except Exception as e:
                Console_UI().warn_user(f"Failed to get size for {self.get_name()}: {e}")

        Console_UI().debug(self.layers)
        self.__set_optimizer()

        self.load()

    def __set_optimizer(self):
        self.__optimizer = None

        if len(list(self.parameters())) > 0:
            self.__optimizer = _get_optimizer(
                type=self.__optimizer_type, parameters=self.parameters()
            )

        self.zero_grad()

    def save(self, scene_name="last"):
        File_Manager().save_pytorch_neural_net(
            self.__neural_net_cfgs.get_clean_name(), self, scene_name
        )

    def get_network_cfgs(self) -> Network_Cfg_Base:
        return self.__neural_net_cfgs

    def get_forward_noise(self):
        if not self.training:
            return 0.0
        if np.random.rand() < 0.5:
            return 0.0
        return Global_Cfgs().forward_noise * np.random.rand()

    def load(self):
        neural_net_name = self.get_name()
        state_dict = File_Manager().load_pytorch_neural_net(
            neural_net_name=neural_net_name
        )
        if state_dict is not None:
            try:
                # As the save is done at the layers level: neural_net.layers.state_dict()
                # we need to load it from the layers
                self.layers.load_state_dict(state_dict)
                Console_UI().debug(f"Loaded state_dict for {neural_net_name}")
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to load dictionary for {neural_net_name} \nError message: {e}"
                )

    def get_name(self):
        return str(self.__neural_net_cfgs)

    def forward_data(self, x):
        x = x.view([-1, *self.__input_shape])
        noise = self.get_forward_noise() * torch.randn_like(x, device=x.device)
        if x.min() >= 0:  # TODO: is the minimum expensive to calculate?
            noise = nn.functional.relu(noise)

        y = self.layers(x + noise)
        return y.view([-1, *self.__output_shape])

    def forward_from_batch(self, batch):
        x = batch[self.__input_name]
        y = self.forward_data(x)

        if self.__output_name in batch:
            # This merge is only happening in morph_visual_morph...
            batch[self.__output_name] = batch[self.__output_name] + y
        else:
            batch[self.__output_name] = y

    def update_learning_rate(self, learning_rate):
        if self.__optimizer:
            if self.__optimizer_type == OptimizerType.Adam:
                _check_learning_rate_adam(learning_rate=learning_rate)
            elif self.__optimizer_type == OptimizerType.SGD:
                _check_learning_rate_SGD(learning_rate=learning_rate)

            print(learning_rate)
            for param_group in self.__optimizer.param_groups:
                param_group["lr"] = learning_rate
                param_group["weight_decay"] = learning_rate / 200

    def reset_optimizer(self):
        self.__set_optimizer()

    def zero_grad(self):
        if self.__optimizer:
            self.__optimizer.zero_grad()

    def step(self):
        if self.__optimizer:
            try:
                self.__optimizer.step()
            except RuntimeError as e:
                Console_UI().warn_user(
                    f"Failed to optimize {self.get_name()} - a masking issue? {e}"
                )

    def update_stochastic_weighted_average_parameters(self):
        """
        This allows us to find a wider local minima
        """
        # Before saving the parameters we remove the backprop gradient
        self.zero_grad()
        self.__weighted_average_parameters_counter += 1

        weight_has_been_updated = False
        if self.__weighted_average_parameters is not None:
            alpha = 1.0 / self.__weighted_average_parameters_counter
            for self_params, prev_params in zip(
                self.parameters(), self.__weighted_average_parameters
            ):
                # TODO: change to numpy or move to cpu
                self_params.data = (1.0 - alpha) * prev_params.to(
                    self_params.data.device
                ) + alpha * self_params.data  # noqa - align ok
            weight_has_been_updated = True

        # Save the updated parameters
        self.__weighted_average_parameters = [
            p.data.clone().detach().cpu() for p in self.parameters()
        ]

        return weight_has_been_updated

    def __check_bn(self, module, flag):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            flag[0] = True

    def check_bn(self):
        flag = [False]
        self.apply(lambda module: self.__check_bn(module, flag))
        return flag[0]

    def reset_bn(self, module):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)

    def __get_momenta(self, module, momenta):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            momenta[module] = module.momentum

    def __set_momenta(self, module, momenta):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momenta[module]

    def prepare_for_batchnorm_update(self):
        # this function is not complete yet

        if not self.check_bn():
            return
        self.train()

        self.apply(self.reset_bn)
        self.apply(lambda module: self.__get_momenta(module, self.__momenta))
        self.__batch_norm_update_counter = 0
        return self.__momenta

    def finish_batchnorm_update(self):
        self.apply(lambda module: self.__set_momenta(module, self.__momenta))

    def update_batchnorm(self, x):
        if isinstance(x, dict):
            batch_size = x["current_batch_size"].item()
        else:
            batch_size = x.shape[0]

        momentum = batch_size / (self.__batch_norm_update_counter + batch_size)
        for module in self.__momenta.keys():
            module.momentum = momentum

        self.__batch_norm_update_counter += batch_size


def _check_learning_rate_SGD(learning_rate: float):
    """Warns if learning rate is outside expected range

    Args:
        learning_rate (float): The learning rate
    """
    if learning_rate > 1e-1:
        time.sleep(10)  # Wait 10 seconds


def _check_learning_rate_adam(learning_rate: float):
    """Warns if learning rate is outside expected range

    Args:
        learning_rate (float): The learning rate
    """
    if learning_rate > 1e-3:
        time.sleep(10)  # Wait 10 seconds


def _get_optimizer(
    type: OptimizerType, parameters: Iterator[torch.nn.Parameter]
) -> torch.optim.Optimizer:
    """
    Choose between optimizers
    """
    if type == OptimizerType.Adam:
        return optim.Adam(parameters, weight_decay=1e-6)

    if type == OptimizerType.SGD:
        return optim.SGD(parameters, lr=1e-2, momentum=0.9, weight_decay=1e-4)

    if type == OptimizerType.ASGD:
        return optim.ASGD(parameters)

    if type == OptimizerType.Adagrad:
        return optim.Adagrad(parameters)

    if type == OptimizerType.Adadelta:
        return optim.Adadelta(parameters)

    if type == OptimizerType.AdamW:
        return optim.AdamW(parameters)

    if type == OptimizerType.RMSprop:
        return optim.RMSprop(parameters)

    raise KeyError(f"Unknown Optimizer type {type}")
