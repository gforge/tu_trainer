from math import ceil
from difflib import get_close_matches
import gc
from typing import Dict, List, Optional, Set

from global_cfgs import Global_Cfgs
from scene import Scene
from DataTypes import Scenario_Cfg, Scene_Cfg_Raw, Scene_Cfg_Extended, Task_Cfg_Raw, \
    Dataset_Modality_Column, Modality_Csv_Mutliple_Columns_Cfg, Modality_Csv_Single_Column_Cfg
from Datasets.helpers import Dictionary_Generator
from UIs.console_UI import Console_UI
from file_manager import File_Manager
from GeneralHelpers import Singleton


class Scenario(metaclass=Singleton):
    """ Collects one or more experiments (scenes).
    """

    def __init__(self, scenario_name: str):
        fm = File_Manager()
        raw_cfgs = fm.read_scenario_config(scenario_name)
        raw_cfgs['name'] = scenario_name
        self.__scenario_cfgs = Scenario_Cfg(**raw_cfgs)

        # In 3.6+ the dictionary are by ordered and thus we don't need OrderedDict
        self.__scene_cfg_dict: Dict[str, Scene_Cfg_Extended] = {}
        for scene_raw in self.scene_definitions:
            extended_args = {k: v for k, v in scene_raw.__dict__.items()}
            extended_args.update(
                tasks={
                    t: Task_Cfg_Raw(**fm.read_task_config(task_name=t, scene_defaults=scene_raw.task_defaults))
                    for t in scene_raw.tasks
                },
                scenario_cfgs=self.__scenario_cfgs,
            )
            self.__scene_cfg_dict[scene_raw.name] = Scene_Cfg_Extended(**extended_args)

        self.__strip_scenes()

        self.__scenario_lengths = [{'name': key, 'len': len(s)} for (key, s) in self.__scene_cfg_dict.items()]

        self.__collect_dictionaries()

        self.__current_scene = None

    @property
    def scenario_name(self):
        return self.__scenario_cfgs.name

    @property
    def scene_definitions(self) -> List[Scene_Cfg_Raw]:
        return self.__scenario_cfgs.scenes

    def __strip_scenes(self):
        start_scene: Optional[str] = Global_Cfgs().get('start_scene')
        if start_scene is None:
            return

        idx = self.find_scene_name_idx(needle=start_scene)
        if idx is None:
            raise ValueError(
                f'Could not find {start_scene} among scene names: {", ".join(self.__scene_cfg_dict.keys())}')

        matched_scene = list(self.__scene_cfg_dict.keys())[idx]
        Console_UI().inform_user(f'Starting at scene no {idx + 1} (\'{matched_scene}\' matches \'{start_scene}\')')
        sliced_scenes: Dict[str, Scene_Cfg_Extended] = {}
        i = 0
        for key in self.__scene_cfg_dict.keys():
            if i >= idx:
                sliced_scenes[key] = self.__scene_cfg_dict[key]
            i += 1
        self.__scene_cfg_dict = sliced_scenes

    def find_scene_name_idx(self, needle):
        needle = needle.strip().lower()
        i = 0
        for scene_name in self.__scene_cfg_dict.keys():
            if scene_name.strip().lower() == needle:
                return i
            i += 1

        # Partial match
        i = 0
        for scene_name in self.__scene_cfg_dict.keys():
            if scene_name.strip().lower().startswith(needle):
                return i
            i += 1

        return None

    def __iter__(self):
        for cfgs in self.__scene_cfg_dict.values():
            gc.collect()
            self.__current_scene = Scene(cfgs=cfgs)

            yield self.__current_scene
            gc.collect()

    def __len__(self):
        return sum([v['len'] for v in self.__scenario_lengths])

    def closing_credits(self):
        Console_UI().inform_user("That's it folks")

    def get_name(self):
        return self.scenario_name

    def get_current_scene(self):
        return self.__current_scene

    def __collect_dictionaries(self):
        """
        Check all the Datasets for common items, e.g. body part and then create
        a general dictionary for all of them.
        """
        datasets: Set[str] = set()
        for scene in self.__scene_cfg_dict.values():
            [datasets.add(task.dataset_name) for task in scene.tasks.values()]

        configs = {}
        for dataset_name in datasets:
            configs[dataset_name] = File_Manager().read_dataset_config(dataset_name)

        modalities_with_dictionaries = [
            'One_vs_Rest',
            'Bipolar',
            'Multi_Bipolar',
        ]  # TODO: add 'hierarchical_label' but this has some fancy logic :-S

        dictionary_candidates = []
        for dataset_name in datasets:
            config = configs[dataset_name]
            try:
                for experiment in config.experiments.values():
                    if isinstance(experiment.modalities, dict):
                        [
                            dictionary_candidates.append(name)
                            for name, cfg in experiment.modalities.items()
                            if cfg.type in modalities_with_dictionaries and name not in dictionary_candidates
                        ]
            except Exception as e:
                raise KeyError(f'Failed to get dictionary for {dataset_name}: {e}')

        # Store all the different values available for this modality into the dictionary singleton that
        # keeps track of the unique values
        dg = Dictionary_Generator()
        for modality_name in dictionary_candidates:
            for dataset_name in datasets:
                dg.append_suggested_dictionary(dataset_name=dataset_name, modality_name=modality_name)

                config = configs[dataset_name]
                for experiment in config.experiments.values():
                    Console_UI().inform_user(f'Reading {dataset_name} file {experiment.annotations_path}')
                    annotations = File_Manager().read_csv_annotations(
                        dataset_name,
                        annotations_rel_path=experiment.annotations_path,
                        # Multi-view argument should be irrelevant for this
                    )
                    if annotations is None:
                        raise ValueError(f'Could not find the dataset: {dataset_name} in {experiment.annotations_path}')

                    modalities = experiment.modalities
                    if modalities == 'same_as_train_set':
                        experiment.modalities = config.experiments['train_set'].modalities

                    if modality_name in modalities:
                        m = modalities[modality_name]
                        if isinstance(m, Modality_Csv_Single_Column_Cfg):
                            try:
                                colname = m.column_name
                                dg.append_values(modality_name=modality_name, values=annotations[colname])
                            except KeyError as e:
                                Console_UI().warn_user(f'Got a key annotation exception for {colname}')
                                Console_UI().warn_user(modalities[modality_name])
                                Console_UI().warn_user(annotations.columns)
                                raise e
                            except Exception as e:
                                Console_UI().warn_user(f'Got an annotation exception for {colname}')
                                Console_UI().warn_user(modalities[modality_name])
                                Console_UI().warn_user(annotations)
                                raise e
                        elif isinstance(m, Modality_Csv_Mutliple_Columns_Cfg):
                            for column_name in m.columns:
                                if isinstance(column_name, Dataset_Modality_Column):
                                    column_name = column_name.csv_name

                                if column_name not in annotations:
                                    n = 3 if len(annotations.columns) < 10 else ceil(len(annotations.columns) / 3)
                                    closest = get_close_matches(
                                        word=column_name,
                                        possibilities=annotations.columns,
                                        n=n,
                                    )
                                    closest = ', '.join(closest)
                                    raise IndexError(f'The {column_name} from {modality_name} doesn\'t exist.' +
                                                     f' Closest matching are: {closest}')

                                dg.append_values(modality_name=modality_name, values=annotations[column_name])
                        else:
                            raise IndexError(f'Expected {modality_name} to have either columns or column_name defined')
