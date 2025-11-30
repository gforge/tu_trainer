import time
from typing import Generic
import torch

from abc import abstractmethod
from .base_loss import Base_Loss, Loss_Cfg_Type


class Base_Convergent_Loss(Generic[Loss_Cfg_Type], Base_Loss[Loss_Cfg_Type]):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.pi_model = self._cfgs.apply.pi_model
        self.classification = self._cfgs.apply.classification
        self.reconstruction = self._cfgs.apply.reconstruction
        self.regression = self._cfgs.apply.regression

        self.has_pseudo_labels = self._cfgs.has_pseudo_labels
        self.pseudo_loss_factor = self._cfgs.pseudo_loss_factor
        self.pseudo_output_name = ''
        if (self.output_name.startswith('encoder_')):
            self.pseudo_output_name = self.output_name.replace('encoder_', 'encoder_pseudo_')

    def forward(self, batch):
        start_time = time.time()

        output, pi_output, target = self.__get_output_and_targets(batch=batch)

        loss = 0

        mod_id = self.modality.get_name()  # Just shorter and easier to read
        if mod_id not in batch['results']:
            batch['results'][mod_id] = {
                'output': output,
                'target': target,
            }

        loss += self.__forward_basic(batch=batch, output=output, target=target)
        loss += self.__forward_regression(batch=batch, output=output, target=target)

        if loss is not None and loss > 0:
            batch['results'][mod_id].update({'loss': loss.item()})

        loss, pseudo_loss = self.__forward_pseudo(batch=batch, mod_id=mod_id, target=target, output=output, loss=loss)
        pi_loss = self.__forward_pi(pi_output, target, batch, mod_id)

        self._forward_analysis(batch, loss)
        batch['time']['process'][self.get_name()] = {'start': start_time, 'end': time.time()}

        return self.__summarize_loss(loss, pi_loss, pseudo_loss)

    def __summarize_loss(self, loss, pi_loss, pseudo_loss):
        total_loss = loss + 0.1 * pi_loss + self.pseudo_loss_factor * pseudo_loss

        # Fix for exploding gradient
        if total_loss > 30:
            total_loss = 0

        return total_loss

    def __get_output_and_targets(self, batch):
        output = batch[self.output_name]
        target = batch[self.target_name]
        output = self.pool_and_reshape_output(output, batch['num_views'])
        target = self.pool_and_reshape_target(target)
        # output.shape = [batch_size, levels] for regular bipolar it is levels=1
        # target.shape = [batch_size]

        pi_output = None
        if isinstance(output, tuple):
            output, pi_output = output

        return output, pi_output, target

    def __forward_basic(self, batch, output, target):
        if not self.classification and not self.reconstruction:
            return 0

        base_loss = self.calculate_loss(output, target)
        batch['loss'][self.get_name()] = base_loss

        return base_loss

    def __forward_regression(self, batch, output, target):
        if not self.regression:
            return 0

        reg_loss = self.calculate_regression_loss(output=output, target=target, batch=batch)
        batch['loss'][self.get_name()] = reg_loss
        return reg_loss

    def __forward_pseudo(self, batch, mod_id, target, output, loss):
        pseudo_loss = 0

        if self.pseudo_output_name not in batch:
            return loss, pseudo_loss

        pseudo_output = self.pool_and_reshape_output(batch[self.pseudo_output_name], batch['num_views'])
        if isinstance(pseudo_output, tuple):
            pseudo_output, _ = pseudo_output

        batch['results'][mod_id].update({
            'pseudo_output': pseudo_output,
        })

        regular_pseudo_loss = self.calculate_loss(pseudo_output, target)
        loss += regular_pseudo_loss

        if self.has_pseudo_labels:
            pseudo_loss += self.calculate_pseudo_loss(pseudo_output, output.detach(), target)

        total_pseudo_loss = pseudo_loss + regular_pseudo_loss
        if isinstance(total_pseudo_loss, torch.Tensor):
            batch['results'][mod_id].update({'pseudo_loss': total_pseudo_loss.item()})

        return loss, pseudo_loss

    def __forward_pi(self, pi_output, target, batch, mod_id):
        pi_loss = 0

        if not self.pi_model or pi_output is None:
            return pi_loss

        pi_loss = self.calculate_pi_loss(pi_output, target)
        if pi_loss > 0:
            batch['results'][mod_id].update({'pi_loss': pi_loss.item()})
        return pi_loss

    def _forward_analysis(self, batch, loss):
        if loss > 0:
            loss = loss.item()

        self.modality.analyze_results(batch, loss)

    @abstractmethod
    def calculate_loss(self, output, target):
        return 0

    def calculate_pi_loss(self, pi_output, target):
        return 0

    def calculate_pseudo_loss(self, pseudo_output, output, target):
        return 0

    def calculate_regression_loss(self, output, target, batch):
        return 0
