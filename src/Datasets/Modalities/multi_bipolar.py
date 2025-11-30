import numpy as np
import pandas as pd

from DataTypes import Dataset_Modality_Column
from DataTypes.enums import LossWeightType
from file_manager import File_Manager
from GeneralHelpers.pytorch_wrapper import wrap

from .bipolar import Bipolar, Modality_Multi_Bipolar_Cfg, calculate_loss_weight


class Multi_Bipolar(Bipolar[Modality_Multi_Bipolar_Cfg]):

    def report_modality_specific_epoch_summary(self, summary):
        summary['modalities'] = self.__report_single_modality(modalities_summary=summary['modalities'],
                                                              runtime_value_name='output',
                                                              prefix='')

        if self.has_runtime_value(name='pseudo_output'):
            summary['modalities'] = self.__report_single_modality(modalities_summary=summary['modalities'],
                                                                  runtime_value_name='pseudo_output',
                                                                  prefix='pseudo_')

    def __report_single_modality(self, modalities_summary, runtime_value_name, prefix):

        for idx, (csv_column, name) in enumerate(self.column_map.items()):
            outputs = self.get_runtime_value(runtime_value_name=runtime_value_name,
                                             subgroup_name=csv_column,
                                             convert_to_numeric=True)

            targets = self.labels.values[:, idx]
            if not self._cfgs.to_each_view_its_own_label:
                targets = targets[self.content.index.get_level_values(1) == 0]
                outputs = outputs[self.content.index.get_level_values(1) == 0]

            performance = self.compute_performance(outputs, targets, prefix=prefix)
            modalities_summary[f'{self.get_name()}_{name}'].update(performance)

        return modalities_summary

    def _prepare_for_report_hook(self, batch):
        (
            all_outputs,
            all_pseudo_outputs,
            all_targets,
            all_indices,
            all_numviews,
        ) = self._get_items_from_report_data(
            items=['output', 'pseudo_output', 'target', 'indices', 'num_views'],
            clear_storage=True,
        )

        results = {}
        for idx, (csv_column, name) in enumerate(self.column_map.items()):
            outputs = all_outputs[:, idx]
            targets = all_targets[:, idx]

            results = self._save_and_analyze_modality_performance(
                output=outputs,
                target=targets,
                indices=all_indices,
                num_views=all_numviews,
                results=results,
                prefix='',
                subgroup_name=csv_column,
            )

            if all_pseudo_outputs is not None:
                pseudo_outputs = all_pseudo_outputs[:, idx]
                results = self._save_and_analyze_modality_performance(
                    output=pseudo_outputs,
                    target=targets,
                    indices=all_indices,
                    num_views=all_numviews,
                    results=results,
                    prefix='pseudo_',
                    subgroup_name=csv_column,
                )

            full_name = f'{self.get_name()}_{name}'
            batch['results'][full_name].update(results)

    def get_item(self, index, num_views=None):
        if self.to_each_view_its_own_label:
            labels = np.array(self.labels.loc[index, :])
        else:
            # All labels are assumed to be identical, hence [0]
            labels = np.array(self.labels.loc[index, 0])
            # Without this all the labels get concatenated into one dimension
            labels = labels.reshape(1, -1)

        return {self.get_batch_name(): labels}

    def prep_content(self):
        assert isinstance(self.content, pd.DataFrame), f'The content should be a dataframe for {self.get_name()}'
        assert len(self.content.columns) > 0, f'There are no columns in the dataframe for {self.get_name()}'
        assert len(self.content) > 0, f'The dataframe is empty for {self.get_name()}'

        self.content = self.content.apply(lambda series: series.apply(lambda x: x.lower()
                                                                      if (isinstance(x, str)) else x))

    def get_num_classes(self):
        return len(self.content.columns)

    def convert_class_names_to_indices(self):
        if self.dictionary is None:
            raise ValueError(f'No dictionary has been generated for {self.get_name()}')

        self.labels = self.content.apply(
            lambda s: s.map(self.cls_name_to_label).fillna(self.ignore_index).astype(np.int32))
        assert len(self.labels.dtypes.unique()) == 1, 'Expected all label types to be of equal type'
        assert self.labels.dtypes[0].kind in ['i', 'u'], 'Expected integer (either signed or unsigned)'

        return self

    @staticmethod
    def get_column_2_name_map(column_defintions, modality_name):
        if column_defintions is None:
            raise ValueError(f'The modality {modality_name} should have a columns attribute')

        columns = {}
        for column in column_defintions:
            if isinstance(column, Dataset_Modality_Column):
                columns[column.csv_name] = column.name
            elif isinstance(column, str):
                columns[column] = column
            else:
                raise ValueError(f'The column defintions {modality_name} can only be string or dict')

        return columns

    @staticmethod
    def get_csv_column_names(column_defintions, modality_name):
        return Multi_Bipolar.get_column_2_name_map(column_defintions=column_defintions,
                                                   modality_name=modality_name).keys()

    @property
    def column_map(self):
        return Multi_Bipolar.get_column_2_name_map(column_defintions=self._cfgs.columns, modality_name=self.get_name())

    @property
    def csv_columns(self):
        return self.column_map.keys()

    def get_csv_column_name(self, name: str):
        for csv, colum_name in self.column_map.items():
            if colum_name == name:
                return csv
        raise IndexError(f'There is no column named {name} in {str(self.column_map)}')

    def collect_statistics(self, labels=None):
        for column in self.csv_columns:
            column_stats = self._get_content_statistics(labels=self.labels[column])
            self.label_stats[column] = column_stats

    def set_loss_weight_type(self, loss_weight_type: LossWeightType):
        if loss_weight_type == self._cfgs.loss_weight_type:
            return

        for column in self.csv_columns:
            column_stats = self.label_stats[column]
            self.label_stats[column]['loss_weight'] = calculate_loss_weight(c=column_stats['label_raw_counter'],
                                                                            loss_weight_type=loss_weight_type)

    def get_loss_weight(self):
        if self._wrapped_weights is None:
            loss_weights = np.array([s['loss_weight'] for s in self.label_stats.values()], dtype=np.float32)
            self._wrapped_weights = wrap(loss_weights)
        return self._wrapped_weights

    def save_dictionary(self):
        if self._cfgs.skip_dictionary_save:
            return

        column_dictionary = pd.DataFrame({
            'columns': [c for c in self.csv_columns],
            'labels': [self.column_map[c] for c in self.csv_columns],
            'index': range(len(self.csv_columns))
        })

        File_Manager().write_dictionary2logdir(dictionary=column_dictionary, modality_name=self.get_name())
