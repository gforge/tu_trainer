from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, List
from Graphs.graph import Graph
from DataTypes import Task_Cfg_Extended, TaskType
from DataTypes.Graph.general import Graph_Cfg_Extended, Graph_Cfg_Raw
from Datasets.experiment_set import Experiment_Set
from Datasets.dataset_factory import Dataset_Factory
from file_manager import File_Manager
from GeneralHelpers import recursive_dict_replace_list_element


class Base_Task(metaclass=ABCMeta):

    def __init__(self, cfgs: Task_Cfg_Extended):
        self._cfgs = cfgs
        self.epoch: int = 0
        self.graphs: Dict[str, Graph] = {}
        self.dataset = Dataset_Factory().get_dataset(dataset_name=self.dataset_name, task_cfgs=self._cfgs)
        self.dataset.set_loss_weight_type(self._cfgs.loss_weight_type)

        self.__init_graphs()

    def __init_graphs(self):
        experiment_names = [self.train_set_name, self.val_set_name]
        if self.test_set_name:
            experiment_names.append(self.test_set_name)

        for experiment_name in experiment_names:
            # Make sure that we have all the expected datasets
            if experiment_name not in self._active_experiment_names:
                raise ValueError(f'The set "{experiment_name}" cannot be found in dataset "{self.dataset_name}"')

            cfgs = self.__get_graph_cfgs(experiment_name=experiment_name)
            experiment = self._get_experiment(experiment_name)
            self.graphs[experiment_name.lower()] = Graph(experiment_set=experiment, cfgs=cfgs)

    @property
    def train_graph(self) -> Graph:
        return self.graphs[self.train_set_name.lower()]

    @property
    def dataset_name(self) -> str:
        "The name/id of the experiement dataset"
        return self._cfgs.dataset_name

    @property
    def _active_experiment_names(self) -> List[str]:
        "The experiments that this task has that are stored under self.graphs"
        return self.dataset.experiments

    def _get_experiment(self, name: str) -> Experiment_Set:
        "Retrieve an experiment object from dataset"
        if name not in self.dataset.experiments:
            raise ValueError(f'The set "{name}" cannot be found in dataset experiment')

        return self.dataset.experiments[name]

    @property
    def task_type(self) -> TaskType:
        "Type of task - currently only training is implemented"
        return self._cfgs.task_type

    @property
    def name(self) -> str:
        return self._cfgs.name

    def get_name(self):
        return self.name

    def __repr__(self) -> str:
        return self.get_name()

    @property
    def graph_name(self) -> str:
        return self._cfgs.graph_name

    def get_scenario_name(self):
        return self.scenario_name

    @property
    def scenario_name(self) -> str:
        "The name of the scenario that this task belongs to"
        return self._cfgs.scenario_name

    @property
    def train_set_name(self) -> str:
        """The training set as defined in Dataset->experiments

        Uses pseudo label set if pseudo labels are active and
        specific pseudo_set_name has been specified

        Returns:
            str: The name of the dataset as defined in the dataset config
        """
        if self._cfgs.has_pseudo_labels:
            return self._cfgs.get('pseudo_set_name', default=self._cfgs.train_set_name)

        return self._cfgs.train_set_name

    @property
    def val_set_name(self) -> Optional[str]:
        """The validation set as defined in Dataset->experiments"""
        return self._cfgs.val_set_name

    @property
    def test_set_name(self) -> Optional[str]:
        """The test set as defined in Dataset->experiments"""
        return self._cfgs.test_set_name

    @abstractmethod
    def save(self, scene_name='last'):
        pass

    def __get_graph_cfgs(self, experiment_name: str) -> Graph_Cfg_Extended:
        base_graph_cfgs = File_Manager().read_graph_config(self.graph_name)
        graph_cfgs_raw = Graph_Cfg_Raw(**base_graph_cfgs)
        fixed_graph_dict = self.__fix_experiment_modalities(graph_cfgs_raw=graph_cfgs_raw,
                                                            experiment_name=experiment_name)
        fixed_graph_dict['task_cfgs'] = self._cfgs
        return Graph_Cfg_Extended(**fixed_graph_dict)

    def __fix_experiment_modalities(self, graph_cfgs_raw: Graph_Cfg_Raw, experiment_name: str) -> Dict[str, Any]:
        exp_set = self._get_experiment(experiment_name)
        replacements = {
            "EXPLICIT_INPUT_MODALITIES": exp_set.get_explicit_input_modality_names(),
            "EXPLICIT_CLASSIFICATION_MODALITIES": exp_set.get_explicit_classification_modality_names(),
            "EXPLICIT_REGRESSION_MODALITIES": exp_set.get_explicit_regression_modality_names(),
            "EXPLICIT_MODALITIES": exp_set.get_explicit_modality_names(),
            "IMPLICIT_INPUT_MODALITIES": exp_set.get_implicit_input_modality_names(),
            "IMPLICIT_CLASSIFICATION_MODALITIES": exp_set.get_implicit_classification_modality_names(),
            "IMPLICIT_REGRESSION_MODALITIES": exp_set.get_implicit_regression_modality_names(),
            "IMPLICIT_MODALITIES": exp_set.get_implicit_modality_names(),
            "EXPLICIT_PSEUDO_OUTPUT_MODALITIES": exp_set.get_explicit_pseudo_output_modality_names(),
            "IMPLICIT_PSEUDO_OUTPUT_MODALITIES": exp_set.get_implicit_pseudo_output_modality_names(),
        }

        graph_dict_values = graph_cfgs_raw.dict()
        for element_id in replacements.keys():
            graph_dict_values = recursive_dict_replace_list_element(source=graph_dict_values,
                                                                    element_id=f'EXPERIMENT_{element_id}',
                                                                    replace_with=replacements[element_id])

        return graph_dict_values
