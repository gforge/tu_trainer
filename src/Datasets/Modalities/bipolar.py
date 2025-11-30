from typing import Generic, TypeVar
import numpy as np
from collections import Counter

from sklearn.metrics import roc_auc_score

from DataTypes import Modality_Bipolar_Cfg, Modality_Multi_Bipolar_Cfg
from DataTypes.enums import LossWeightType

from .Base_Modalities.helpers.dim import fix_dims
from .Base_Modalities.base_label import Base_Label
from .Base_Modalities.base_output import Base_Output
from UIs.console_UI import Console_UI
from Graphs.Losses.bipolar_margin_loss import Bipolar_Margin_Loss


def compute_auc(outputs, targets):
    mask = np.logical_or(targets == 1, targets == -1)
    if len(mask) < 2 or mask.sum() < 2:
        return np.nan

    try:
        auc = roc_auc_score(y_true=targets[mask] == 1, y_score=outputs[mask])
    except IndexError as error:
        # TODO: Why is this throwing?
        msg = f'IndexError in AUC calculation: {error}'
        Console_UI().warn_user(msg)
        return np.nan
    except ValueError as error:
        msg = f'ValueError in AUC calculation: {error}'
        Console_UI().warn_user(msg)
        return np.nan

    return auc


def calculate_loss_weight(c: Counter, loss_weight_type: LossWeightType) -> np.ndarray:
    """
    Set the self.label_stats['loss_weight'] to a weight that can be used to multiply the
    loss function with.
    """
    if loss_weight_type == LossWeightType.max:
        # Limit the loss to max 1/2:th of the total when there are > 2000 observations
        # We've found that using np.sqr or smaller limitations causes strange collapses
        # in the multi-bipolar estimates
        total_with_label = sum([c[i] for i in [-1, 1]])
        max_for_loss = max(1000, total_with_label / 3)
        loss_weight = np.array([1 / min(c[i] + 1, max_for_loss) for i in [-1, 1]])

    elif loss_weight_type == LossWeightType.sqrt:
        # Square root loss
        loss_weight = np.array([1 / np.sqrt(c[i] + 1) for i in [-1, 1]])

    elif loss_weight_type == LossWeightType.basic:
        # Basic loss
        loss_weight = np.array([1 / (c[i] + 1) for i in [-1, 1]])

    else:
        raise ValueError(f'Invalid loss weight type: {loss_weight_type}')

    # Normalize the weights so that sum equals 1
    return loss_weight / np.sum(loss_weight)


BipolarCfg = TypeVar('BipolarCfg', Modality_Multi_Bipolar_Cfg, Modality_Bipolar_Cfg)


class Bipolar(Generic[BipolarCfg], Base_Label[BipolarCfg], Base_Output):

    def compute_performance(self, outputs, targets, prefix: str):
        outputs = fix_dims(outputs)
        targets = fix_dims(targets)
        outputs = outputs.reshape(targets.shape)

        positive_mask = targets == 1
        negative_mask = targets == -1
        true_positive = np.sum(outputs[positive_mask] > 0)
        false_positive = np.sum(outputs[negative_mask] > 0)
        false_negative = np.sum(outputs[positive_mask] < 0)
        true_negative = np.sum(outputs[negative_mask] < 0)

        sensitivity = np.nan
        specificity = np.nan
        precision = np.nan
        negative_predictive_value = np.nan
        auc = np.nan

        if np.any(positive_mask):
            sensitivity = true_positive / positive_mask.sum()
        if np.any(negative_mask):
            specificity = true_negative / negative_mask.sum()
        if (true_positive + false_positive) != 0:
            precision = true_positive / (true_positive + false_positive)
        if (true_negative + false_negative) != 0:
            negative_predictive_value = true_negative / (true_negative + false_negative)

        # AUC can't be calculated if there is only one group
        if np.any(positive_mask) and np.any(negative_mask):
            auc = compute_auc(outputs=outputs, targets=targets)

        accuracy = self.compute_accuracy(outputs, targets)
        accuracy = self.get_mean_accuracy(accuracy)

        return {
            f'{prefix}sensitivity': sensitivity,
            f'{prefix}specificity': specificity,
            f'{prefix}precision': precision,
            f'{prefix}negative_predictive_value': negative_predictive_value,
            f'{prefix}auc': auc,
            f'{prefix}accuracy': accuracy,
            # f'{prefix}kappa': kappa, - not really using...
        }

    def report_modality_specific_epoch_summary(self, summary):
        outputs = self.get_runtime_value('output', convert_to_numeric=True)
        targets = self.labels.values

        performance = self.compute_performance(outputs, targets, prefix='')
        summary['modalities'][self.get_name()].update(performance)

        if not self.has_runtime_value('pseudo_output'):
            return
        pseudo_outputs = self.get_runtime_value('pseudo_output', convert_to_numeric=True)

        performance = self.compute_performance(pseudo_outputs, targets, prefix='pseudo_')
        summary['modalities'][self.get_name()].update(performance)

    def analyze_modality_specific_results(self, batch):
        result_keys = ('output', 'pseudo_output', 'target')
        base_keys = ('indices', 'num_views')

        # Save for batching multiple analysis specific for the output step
        for key in (*result_keys, *base_keys):
            if key in result_keys:
                if key not in batch['results'][self.get_name()]:
                    continue

                value = batch['results'][self.get_name()].pop(key)
            else:
                value = batch[key]

            self._add_2_report_data(key=key, value=value)
        batch['results_hooks'].add(self._prepare_for_report_hook)

    def _prepare_for_report_hook(self, batch):
        (
            output,
            pseudo_output,
            target,
            all_indices,
            all_numviews,
        ) = self._get_items_from_report_data(
            items=['output', 'pseudo_output', 'target', 'indices', 'num_views'],
            clear_storage=True,
        )

        results = {}

        results = self._save_and_analyze_modality_performance(
            output=output,
            target=target,
            indices=all_indices,
            num_views=all_numviews,
            results=results,
            prefix='',
        )

        if pseudo_output is not None:
            results = self._save_and_analyze_modality_performance(
                output=pseudo_output,
                target=target,
                indices=all_indices,
                num_views=all_numviews,
                results=results,
                prefix='pseudo_',
            )

        batch['results'][self.get_name()].update(results)

    def _save_and_analyze_modality_performance(self,
                                               output,
                                               target,
                                               indices,
                                               num_views,
                                               results,
                                               prefix,
                                               subgroup_name=None):
        output = output.reshape(-1)
        target = target.reshape(-1)
        self.set_runtime_value(f'{prefix}output',
                               value=output,
                               indices=indices,
                               num_views=num_views,
                               subgroup_name=subgroup_name)

        if self._cfgs.calculate_entropy:
            entropy = self.compute_entropy(output=output)
            self.set_runtime_value(f'{prefix}entropy',
                                   value=entropy,
                                   indices=indices,
                                   num_views=num_views,
                                   subgroup_name=subgroup_name)

        accuracy = self.compute_accuracy(output=output, target=target)
        results[f'{prefix}accuracy'] = self.get_mean_accuracy(accuracy)

        return results

    def compute_entropy(self, output):
        return np.exp(-(output**2))

    def compute_accuracy(self, output, target):
        # For some reason the dimension can get lost and we get only a single value
        # TODO: fix output dim lost
        output = fix_dims(output)
        target = fix_dims(target)

        accuracy = np.ones(target.shape) * self.ignore_index
        accuracy[target == 1] = output[target == 1] > 0
        accuracy[target == -1] = output[target == -1] < 0
        return accuracy

    def collect_statistics(self, labels=None):
        self.label_stats.update(self._get_content_statistics())

    def _get_content_statistics(self, labels=None):
        if labels is None:
            labels = self.labels[self.labels != self.ignore_index]

        if not self._cfgs.to_each_view_its_own_label:
            # labels = [l.to_list()[0] for idx, l in labels.groupby(level=0)]
            # Better performance:
            labels = labels[labels.index.get_level_values(level=1) == 0]

        statistics = {}
        c = Counter(labels)
        statistics['labels'] = np.array([-1, 0, 1])
        statistics['label_raw_counter'] = c
        statistics['label_count'] = np.array([c[-1], c[0], c[1]])
        statistics['label_likelihood'] = statistics['label_count'] / np.sum(statistics['label_count'])
        statistics['num_classes'] = self.get_num_classes()
        statistics['loss_weight'] = calculate_loss_weight(c=c, loss_weight_type=self._cfgs.loss_weight_type)

        label_informativeness = {}
        for label_index, label_likelihood in zip(statistics['labels'], statistics['label_likelihood']):
            label_informativeness[label_index] = (1 - label_likelihood) * self.signal_to_noise_ratio

        statistics['label_informativeness'] = label_informativeness

        return statistics

    def set_loss_weight_type(self, loss_weight_type: LossWeightType):
        if loss_weight_type == self._cfgs.loss_weight_type:
            return

        self.label_stats['loss_weight'] = calculate_loss_weight(c=self.label_stats['label_raw_counter'],
                                                                loss_weight_type=loss_weight_type)

    def get_num_classes(self):
        return 1

    def get_loss_type(self):
        return Bipolar_Margin_Loss.__name__.lower()
