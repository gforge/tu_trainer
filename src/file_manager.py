from __future__ import annotations
import os
import json
from difflib import get_close_matches
from pathlib import Path
from torch.nn import Module
from yaml import load as yamlLoad, Loader as YamlLoader, dump as yamlDump
import re
import numpy as np
import pandas as pd
import torch
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from typing import TYPE_CHECKING, Dict, Any, List, Optional
from pydantic import ValidationError

from DataTypes import Dataset_Cfg_Raw, Modality_Cfg_Parser
from GeneralHelpers import Singleton
from UIs.Writers import WriterLibrary
from UIs.console_UI import Console_UI

if TYPE_CHECKING:
    from Graphs.Factories.Networks.neural_net import Neural_Net

np.set_printoptions(precision=3, linewidth=100000, threshold=10000)


def get_base_dir(path: str):
    if os.path.isdir(path):
        return path

    super_path = re.sub("/[^/]+$", "", path)
    if len(super_path) == 0 or path == super_path:
        return None

    return get_base_dir(super_path)


_valid_def_file_regex = "\\.(json|yaml)$"


class File_Manager(metaclass=Singleton):

    def __init__(
        self,
        annotations_root,
        scenario_log_root,
        tmp_root,
        model_zoo_root: str,
        resume_prefix,
        resume_scene,
        log_folder,
        global_cfgs,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.annotations_root = annotations_root
        self.scenario_log_root = scenario_log_root
        self.tmp_root = tmp_root
        self.__init_model_zoo(model_zoo_root=model_zoo_root)
        self.resume_prefix = resume_prefix
        self.resume_scene = resume_scene

        self.log_folder = log_folder
        WriterLibrary().set_log_folder(self.log_folder)
        self.log_configs = True

        self.__graph_folder = os.path.join('..', 'configs', 'Graphs')
        self.__dataset_folder = os.path.join('..', 'configs', 'Datasets')

        # TODO - set iteration counter on init to last value (this should be saved in a iteration counter txt file)

        # We can't use the Singleton pattern here as the Global_Cfgs() imports and initiates File_Manager
        self.global_cfgs = global_cfgs

        self.__cache = {}

    def __init_model_zoo(self, model_zoo_root: Optional[str]) -> None:
        self.model_zoo_root = model_zoo_root
        if self.model_zoo_root is None:
            Console_UI().inform_user('No model zoo path set')
            return

        if not os.path.isdir(self.model_zoo_root):
            raise FileNotFoundError(f'The path {model_zoo_root} for model zoo is not a valid directory')

        hub_model_dir = os.path.join(self.model_zoo_root, 'hub_models')
        if not os.path.isdir(hub_model_dir):
            os.mkdir(hub_model_dir)
        torch.hub.set_dir(hub_model_dir)

    def get_annotations_path(self, dataset_name: str):
        return os.path.join(self.annotations_root, dataset_name.lower())

    def read_graph_config(self, graph_name: str) -> Dict[str, Any]:
        config_path = os.path.join(self.__graph_folder, graph_name)
        graph_config = self.__read_config(config_path)
        if graph_config is None:
            msg = _retrieve_error_txt_with_similar_filenames(folder=self.__graph_folder,
                                                             needle=graph_name,
                                                             valid_file_endings=['.yaml', '.json'])
            raise FileNotFoundError(msg)
        return graph_config

    def read_dataset_config(self, dataset_name: str):
        config_path = os.path.join(self.__dataset_folder, dataset_name)
        dataset_cfgs = self.__read_config(config_path)
        if dataset_cfgs is None:
            msg = _retrieve_error_txt_with_similar_filenames(folder=self.__dataset_folder,
                                                             needle=dataset_name,
                                                             valid_file_endings=['.csv'])
            raise FileNotFoundError(msg)
        dataset_cfgs = Dataset_Cfg_Raw(**dataset_cfgs)
        dataset_cfgs = self.__resolve_shared_cfgs(dataset_cfgs=dataset_cfgs)
        return dataset_cfgs

    def __resolve_shared_cfgs(self, dataset_cfgs: Dict[str, Any]) -> Dataset_Cfg_Raw:
        """
        As we do not want to have duplicated definitions that we maintain we
        keep similar configs under Datasets/- shared/. This allows us to
        change all definitions at once so that we can avoid having
        definitions that are not fully matching.
        """
        experiments_with_shared = [
            e for e in dataset_cfgs.experiments.values() if e.get('shared_modalities') is not None
        ]
        for experiment_cfgs in experiments_with_shared:
            for modality_name in experiment_cfgs.get('shared_modalities'):
                config_path = os.path.join('..', 'configs', 'Datasets', 'shared', modality_name)
                config = self.__read_config(config_path)
                try:
                    shared_definition = Modality_Cfg_Parser.model_validate(config).root
                except ValidationError as e:
                    raise ValueError(f'Failed to handle {config} in {config_path}, got: {e}')
                experiment_cfgs.modalities[modality_name] = shared_definition

        return dataset_cfgs

    def get_dataset_definitions(self):
        config_path = os.path.join('..', 'configs', 'Datasets')
        return [
            re.sub(_valid_def_file_regex, "", fn)
            for fn in os.listdir(config_path)
            if re.search(_valid_def_file_regex, fn)
        ]

    def read_scenario_config(self, scenario_name):
        if self.resume_prefix is not None and self.global_cfgs.get('resume_config', False):
            print(f'Resuming scenario configs {scenario_name} from {self.resume_prefix}')
            scenario_cfgs = self.__read_config(os.path.join(self.scenario_log_root, self.resume_prefix, 'scenario'))
            if (scenario_cfgs is not None):
                return scenario_cfgs

        config_dir = os.path.join('..', 'configs', 'Scenarios')
        config_path = os.path.join(config_dir, '%s' % (scenario_name.lower()))

        scenario = self.__read_config(config_path)
        if scenario is None:
            available_configs = [
                f for f in os.listdir(config_dir)
                if os.path.isfile(os.path.join(config_dir, f)) and re.match(_valid_def_file_regex, f)
            ]
            raise ImportError("Could not load file: %s files in that dir: \n - %s" %
                              (config_path, '\n - '.join(available_configs)))
        return scenario

    def get_task_config_path(self, path):
        config_path = os.path.join('..', 'configs', 'Tasks')
        for p in path.split('/'):
            config_path = os.path.join(config_path, p)
        return config_path

    def __build_task(self, task: Dict, scene_defaults: Dict = {}) -> Dict:

        def merge_tasks(master: Dict, slave: Dict) -> Dict:
            merged_apply = {**slave.get('apply', {}), **master.get('apply', {})}
            return {**slave, **master, 'apply': merged_apply}

        # Add general template definitions
        if 'template' in task:
            template = self.read_task_config(task_name=f'Template/{task["template"]}')
            task = merge_tasks(master=task, slave=template)

        # Scene defaults overridde task definitions
        if isinstance(scene_defaults, dict):
            task = merge_tasks(master=scene_defaults, slave=task)

        return task

    def read_task_config(self, task_name, scene_defaults: Dict = {}):
        # Convert Ankle/regulated to full path including the yaml
        config_path = self.get_task_config_path(path=task_name)

        task = self.__read_config(config_path, log_prefix="Task_")
        if task is None:
            config_base = get_base_dir(config_path)
            available_configs = [
                f for f in os.listdir(config_base)
                if os.path.isfile(os.path.join(config_base, f)) and re.match(_valid_def_file_regex, f)
            ]
            raise ImportError("Could not load file: %s files in that dir: \n - %s" %
                              (config_path, '\n - '.join(available_configs)))

        return self.__build_task(task=task, scene_defaults=scene_defaults)

    def __read_config(self, path: str, log_prefix: Optional[str] = None):
        if not re.match(_valid_def_file_regex, path):
            if os.path.exists(f'{path}.json'):
                path = f'{path}.json'
            elif os.path.exists(f'{path}.yaml'):
                path = f'{path}.yaml'

        pathObj = Path(path)
        config_data = None
        # The idea with none it could have a fallback default
        if pathObj.is_file():
            with open(pathObj, 'r') as fstr:
                if re.match(".+\\.json$", path):
                    config_data = json.load(fstr)
                else:
                    config_data = yamlLoad(fstr, Loader=YamlLoader)

        if self.log_configs and config_data is not None:
            self.log_setup(data=config_data, name=pathObj, log_prefix=log_prefix)

        return config_data

    def log_setup(self, data: dict, name: Path, log_prefix: Optional[str] = None):
        fn = name.name
        if not fn.endswith('.yaml'):
            if fn.endswith('.json'):
                fn = fn[:-len('json')] + 'yaml'
            else:
                fn = f'{fn}.yaml'

        # Scenario & Dataset configs may have the same name and
        # therefore we need to prefix according to the last dirname
        if name.parent.name not in ['', '.']:
            fn = f'{name.parent.name}_{fn}'

        if log_prefix is not None:
            fn = f'{log_prefix}{fn}'

        dest_cfg_log_fn = os.path.join(self.log_folder, 'setup', fn)
        if not os.path.exists(dest_cfg_log_fn):
            self.make_sure_dir_exist(file_path=dest_cfg_log_fn)
            with open(dest_cfg_log_fn, 'w') as fstr:
                yamlDump(data, stream=fstr, width=1000)

    def get_available_csvs(self, dataset_name: str):
        path = self.get_annotations_path(dataset_name)
        return [i[:-len('.csv')] for i in os.listdir(path) if re.search("\\.csv$", i)]

    def read_csv_annotations(
        self,
        dataset_name: str,
        annotations_rel_path: str,
        multi_view_per_sample: bool = False,
    ):
        annotations_path = os.path.join(self.get_annotations_path(dataset_name), annotations_rel_path)

        if os.path.exists(annotations_path):
            cache_path = f'csv:{annotations_path}'
            if cache_path in self.__cache:
                annotation = self.__cache[cache_path].copy()
            else:
                if self.global_cfgs.get('test_run', default=False):
                    annotation = pd.read_csv(annotations_path, low_memory=False, nrows=1e4)
                else:
                    annotation = pd.read_csv(annotations_path, low_memory=False)
                self.__cache[cache_path] = annotation

            if multi_view_per_sample and not isinstance(annotation.index, pd.MultiIndex):
                if 'index' not in annotation:
                    annotation['index'] = np.arange(len(annotation), dtype=int)
                else:
                    assert np.issubdtype(annotation['index'], np.dtype(int)), 'Index should be integers'
                    assert annotation['index'].min() == 0, 'The index has to be indexed from 0'

                if 'sub_index' not in annotation:
                    annotation['sub_index'] = np.zeros(len(annotation), dtype=int)
                else:
                    assert np.issubdtype(annotation['sub_index'], np.dtype(int)), 'Sub index should be integers'
                    assert annotation['sub_index'].max() > 0, 'You have provided a sub_index without purpose (max 0)'
                    assert annotation['sub_index'].min() == 0, 'The sub_index has to start from 0'

                annotation.set_index(['index', 'sub_index'], inplace=True)
            if 'num_views' not in annotation:
                annotation['num_views'] =\
                    [a for b in [[i] * i for i in annotation.groupby(level=0).size()] for a in b]

            return annotation

        Console_UI().warn_user(f'Failed to load file from disk: \'{annotations_path}\'')
        return None

    def write_csv_annotation(
        self,
        annotations: pd.DataFrame,
        dataset_name: str,
        experiment_file_name: str,
    ):
        annotations_path = os.path.join(self.log_folder, dataset_name, experiment_file_name)
        self.make_sure_dir_exist(annotations_path)
        Console_UI().inform_user(f'Save {dataset_name} to {experiment_file_name}')
        annotations.to_csv(annotations_path, index=True)

    def read_dictionary(
        self,
        dataset_name: str,
        modality_name: str,
    ):
        """
        If we have a dictionary associated with the current weights we should use those.
        The fallback is the resume weight's dictionary and lastly the annotation's dictionary.
        """
        filename = f'{modality_name.lower()}_dictionary.csv'
        cachename = f'dictionary:{dataset_name}->{filename}'
        if cachename in self.__cache:
            return self.__cache[cachename]

        dictionary_path = os.path.join(self.log_folder, 'neural_nets', filename)

        if (not os.path.exists(dictionary_path) and self.resume_prefix is not None):
            dictionary_path = os.path.join(self.scenario_log_root, self.resume_prefix, 'neural_nets', filename)

        if not os.path.exists(dictionary_path):
            dictionary_path = os.path.join(self.get_annotations_path(dataset_name), filename)

        if os.path.exists(dictionary_path):
            try:
                dictionary = pd.read_csv(dictionary_path)
                self.__cache[cachename] = dictionary
                return dictionary
            except pd.errors.EmptyDataError:
                Console_UI().warn_user(f'The dictionary for {modality_name} is corrupt - see file {dictionary_path}')

        return None

    def write_dictionary(
        self,
        dictionary: pd.DataFrame,
        dataset_name: str,
        modality_name: str,
    ):
        """
        We save the dictionary with the annotations and the network weights. If we resume the weights
        then it is critical that the weights are interpreted with the same dictionary as used for those
        weights in case the order of the items change
        """
        filename = f'{modality_name.lower()}_dictionary.csv'
        dictionary_path = os.path.join(self.get_annotations_path(dataset_name), filename)

        self.make_sure_dir_exist(dictionary_path)
        dictionary.to_csv(dictionary_path, index=False)

        self.write_dictionary2logdir(dictionary=dictionary, modality_name=modality_name)

    def write_dictionary2logdir(
        self,
        dictionary: pd.DataFrame,
        modality_name: str,
    ):
        filename = f'{modality_name.lower()}_dictionary.csv'
        neural_net_dictionary_path = os.path.join(self.log_folder, 'neural_nets', filename)
        self.make_sure_dir_exist(neural_net_dictionary_path)
        dictionary.to_csv(neural_net_dictionary_path, index=False)

    def write_usage_profile(self, scene_name: str, task: str, memory_usage: pd.DataFrame):
        memory_path = os.path.join(self.log_folder, f'{scene_name}_{task.replace("/", "_")}_memory_usage.csv')
        memory_usage.to_csv(memory_path, index=True)

    def load_and_save_hub_network(self, repo_or_dir: str, model: str, pretrained: bool) -> Module:
        assert self.model_zoo_root is not None, \
            'You must set the model zoo directory when trying to use pretrained models'

        Console_UI().inform_user(f'Loading from hub model {model}@{repo_or_dir}')
        loaded_model = torch.hub.load(repo_or_dir=repo_or_dir, model=model, pretrained=pretrained)

        Console_UI().inform_user(f'Done loading {model}@{repo_or_dir}')
        return loaded_model

    def load_pytorch_neural_net(self, neural_net_name: str):
        current_run_last_save = self.get_network_full_path(neural_net_name=neural_net_name, scene_name='last')
        if os.path.exists(current_run_last_save):
            Console_UI().inform_user(f'Resuming current runs {neural_net_name:>90}::last network')
            return torch.load(current_run_last_save)['state_dict']

        if self.resume_prefix is not None:
            scene_name = self.global_cfgs.get('resume_scene')
            network_filename = self.get_network_filename(neural_net_name=neural_net_name, scene_name=scene_name)
            neural_net_path = os.path.join(self.scenario_log_root, self.resume_prefix, 'neural_nets', network_filename)
            if os.path.exists(neural_net_path):
                Console_UI().inform_user(f'Resuming from {self.resume_prefix} the network {network_filename}')
                return torch.load(neural_net_path)['state_dict']

        if not self.global_cfgs.get('silent_init_info'):
            Console_UI().inform_user(f'{neural_net_name} does not exist, Initializing from scratch')

        return None

    def save_pytorch_neural_net(
        self,
        neural_net_name: str,
        neural_net: Neural_Net,
        scene_name: str = 'last',
        full_network=False,
    ):
        neural_net_path = self.get_network_full_path(neural_net_name=neural_net_name, scene_name=scene_name)
        self.make_sure_dir_exist(neural_net_path)
        # This prints too much stuff to realy be useful
        # Console_UI().inform_user(f'Saving {neural_net_name:>90}::{scene_name} to: {os.path.dirname(neural_net_path)}')
        torch.save(
            {
                'state_dict': neural_net.layers.state_dict(),
                'neural_net_cfgs': neural_net.get_network_cfgs().dict()
            }, neural_net_path)

    def get_network_filename(self, neural_net_name: str, scene_name: str):
        return f'{neural_net_name}_{scene_name}.t7'

    def get_network_dir_path(self):
        return os.path.join(self.log_folder, 'neural_nets')

    def get_network_full_path(self, neural_net_name: str, scene_name: str):
        network_filename = self.get_network_filename(neural_net_name=neural_net_name, scene_name=scene_name)
        return os.path.join(self.get_network_dir_path(), network_filename)

    def write_description(
        self,
        file_path,
        description: str,
    ):
        self.make_sure_dir_exist(file_path)
        desc_file = open(file_path, 'w')
        desc_file.write(description)
        desc_file.close()

    def make_sure_dir_exist(self, file_path: str):
        dir_name = os.path.dirname(file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

    def download_zip_file(self, url, dataset_name):
        """
        annotations has to be zipped with the following command:
        7z a -ppassword -mem=ZipCrypto imagenet.zip imagenet
        """

        url_content = urlopen(url)
        zipfile = ZipFile(BytesIO(url_content.read()))

        pswd = Console_UI().receive_password('Password for unzipping annotations of %s dataset:' % (dataset_name))
        zipfile.extractall(self.annotations_root, pwd=bytes(pswd, 'utf-8'))


def _retrieve_error_txt_with_similar_filenames(folder: str, needle: str, valid_file_endings: List[str]) -> str:

    def __check_file_ending(filename: str) -> bool:
        for v in valid_file_endings:
            if filename.endswith(v):
                return True
        return False

    available_files = [f for f in os.listdir(folder) if __check_file_ending(f)]
    if len(available_files) == 0:
        return f'Could not find {needle} or anything similar in {os.path.abspath(folder)}'

    # Throw error as we failed to find the definition
    closest_match = get_close_matches(word=needle, possibilities=available_files)
    if len(closest_match) == 0:
        matches = "\n - " + "\n - ".join(available_files)
        return f'Could not find {needle} or anything similar in {os.path.abspath(folder)} among files: {matches}'

    matches = "\n - " + "\n - ".join(closest_match)
    return f'Could not find {needle}, closest found match: {matches}'
