from typing import Dict

from DataTypes.Dataset.extended import Dataset_Cfg_Extended
from DataTypes.Dataset.raw import Dataset_Cfg_Raw
from DataTypes.Task.extended import Task_Cfg_Extended
from file_manager import File_Manager
from GeneralHelpers import Singleton
from .csv_dataset import CSV_Dataset as Dataset


class Dataset_Factory(metaclass=Singleton):

    def __init__(self):
        self.datasets: Dict[str, Dataset] = {}

    def get_dataset(self, dataset_name, task_cfgs: Task_Cfg_Extended) -> Dataset:
        fixed_name = dataset_name.lower()
        if fixed_name not in self.datasets:
            self.datasets[fixed_name] = self.__build_new_dataset(dataset_name=fixed_name, task_cfgs=task_cfgs)
        else:
            self.datasets[fixed_name].set_batch_size_multiplier(task_cfgs.batch_size_multiplier)
            self.datasets[fixed_name].set_view_dropout(task_cfgs.view_dropout)

        return self.datasets[fixed_name]

    def __build_new_dataset(self, dataset_name: str, task_cfgs: Task_Cfg_Extended):
        predefined_datasets = File_Manager().get_dataset_definitions()
        if dataset_name not in predefined_datasets:
            all_sets = '", "'.join(predefined_datasets)
            raise IndexError(f'The dataset "{dataset_name}" is not among the predefined sets: "{all_sets}"')

        base_configs = _get_dataset_cfgs(dataset_name=dataset_name).dict()
        base_configs.update(task_cfgs=task_cfgs, name=dataset_name)
        cfgs = Dataset_Cfg_Extended(**base_configs)

        return Dataset(cfgs=cfgs)


def _get_dataset_cfgs(dataset_name) -> Dataset_Cfg_Raw:
    dataset_cfgs = File_Manager().read_dataset_config(dataset_name)
    dataset_cfgs = _copy_same_as_cfgs(dataset_cfgs=dataset_cfgs)
    return dataset_cfgs


def _copy_same_as_cfgs(dataset_cfgs: Dataset_Cfg_Raw) -> Dataset_Cfg_Raw:
    """
    For the sake of simplicity, when modalities are identical during
    train and tests, we can just write "modalities": "same_as_X" in
    the config file.(in this example, X is "train")

    This function searches for the modalities like this and
    replace them with the "X" modalities
    """
    for experiment_cfgs in dataset_cfgs.experiments.values():
        if isinstance(experiment_cfgs.modalities, str) and experiment_cfgs.modalities.startswith('same_as_'):
            other_experiment = experiment_cfgs.modalities[len('same_as_'):]
            experiments = dataset_cfgs.experiments
            if other_experiment in experiments:
                experiment_cfgs.modalities = experiments[other_experiment].modalities
            else:
                all_experiments = '\', \''.join(experiments.keys())
                raise KeyError(f'Could not find the modality \'{other_experiment}\'' +
                               f' among the modalities: \'{all_experiments}\'')

    return dataset_cfgs
