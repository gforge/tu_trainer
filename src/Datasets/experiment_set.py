from __future__ import annotations
import traceback
import math
import time
import pandas as pd
import numpy as np
from torch import Tensor
from collections import defaultdict
from typing import Dict, Any, Iterable, Optional, Union, List
from pydantic import ValidationError

from DataTypes import Modality_Cfg_Parser, Dataset_Experiment_Cfg_Extended
from DataTypes.Modality.other import Modality_Cfg_Base
from DataTypes.enums import LossWeightType
from Datasets.Modalities.bipolar import Bipolar
from Datasets.Modalities.multi_bipolar import Multi_Bipolar

from .helpers import get_modality_and_content, Any_Modality
from GeneralHelpers import collate_factory, move_2_cuda_if_gpu_activated, gpu_activated, unwrap
from .Modalities.Base_Modalities.base_runtime_value import Base_Runtime_Value
from torch.utils.data import DataLoader, Dataset
from file_manager import File_Manager
from global_cfgs import Global_Cfgs
from UIs.console_UI import Console_UI
from .batch_loader import Batch_Loader


def worker_init_fn(iterator_no: int):

    def inner_worker_init_fn(worker_id: int):
        np.random.seed(np.random.get_state()[1][0] + worker_id + iterator_no * 100)

    return inner_worker_init_fn


class Experiment_Set(Dataset):

    def __init__(self, cfgs: Dataset_Experiment_Cfg_Extended, save_annotations_filename: str):
        self.__cfgs = cfgs

        self.__explicit_modalities: Dict[str, Any_Modality] = {}
        self.__implicit_modalities: Dict[str, Any_Modality] = {}
        self.__modalities: Dict[str, Any_Modality] = {}

        self.__explicit_input_modalities: Dict[str, Any_Modality] = {}
        self.__explicit_output_modalities: Dict[str, Any_Modality] = {}

        self.__batch_size_multiplier: int = -1
        self.__view_dropout: int = -1
        self.set_batch_size_multiplier(number=self.__cfgs.batch_size_multiplier)
        self.set_view_dropout(view_dropout=self.__cfgs.view_dropout)

        self.__setup_annotations()
        self.__setup_modalities()
        self.__dataloader_iterator: Optional[DataLoader] = None
        self.reset_epoch()

        self.__save_annotations_filename = save_annotations_filename
        self.__no_iterators = 0

    @property
    def ignore_index(self) -> int:
        return self.__cfgs.ignore_index

    @property
    def __dataset_name(self):
        return self.__cfgs.dataset_name

    @property
    def name(self):
        return self.__cfgs.name

    def __repr__(self) -> str:
        return f'{self.__dataset_name} >> {self.name}'

    def set_batch_size_multiplier(self, number: float) -> Experiment_Set:
        assert number > 0, 'The batch size multiplier must be greater than 0 as we can\'t have negative sized batches'
        # First run the multiplier has an impossible value
        if self.__batch_size_multiplier == -1:
            self.__batch_size_multiplier = number

        if self.__batch_size_multiplier == number:
            return self

        # Rebuild the batches once we have a new batch size
        self.__batch_size_multiplier = number
        self.reset_epoch()

        return self

    def set_view_dropout(self, view_dropout: Optional[float]) -> Experiment_Set:
        """
        Updates the view dropout at runtime for each of the experiments.

        The dropout controlles how many images will be used at each step
        from the total. 0 or None indicates that all will be used.
        """
        assert view_dropout is None or (view_dropout > 0 and view_dropout <= 1), \
            'The view dropout must be None or a proportion, i.e. between 0 and 1'

        # First run the dropout has an impossible value
        if self.__view_dropout == -1:
            self.__view_dropout = view_dropout

        if self.__view_dropout == view_dropout:
            return self

        if self.__view_dropout != view_dropout:
            # Rebuild the batches once we have a new view dropout proportion
            self.__view_dropout = view_dropout
            self.reset_epoch()

        return self

    def set_loss_weight_type(self, loss_weight_type: LossWeightType) -> Experiment_Set:
        """
        Updates the loss weight type at runtime for each of the experiments.
        """
        for modality in self.__modalities.values():
            if isinstance(modality, (Bipolar, Multi_Bipolar)):
                modality.set_loss_weight_type(loss_weight_type)

        return self

    @property
    def batch_size(self) -> int:
        return self.__cfgs.batch_size

    def __reset_iterator(self) -> None:
        self.__dataloader_iterator = None

    def __iter__(self):
        # When a for loop calls for the iterator it should be reset
        self.__reset_iterator()
        return self

    def __len__(self):
        return len(self.__get_iterator())

    def __next__(self):
        batch = None
        i = 0
        iterator = self.__get_iterator()
        while (batch is None and i < 5):
            try:
                batch = next(iterator)
                if batch['encoder_image'].max() == 0:
                    raise ValueError('No non-zero images in batch - check file folder')

                # Convert all tensors to cuda if environment calls for it
                for key in batch:
                    batch[key] = move_2_cuda_if_gpu_activated(batch[key])

                if isinstance(batch['num_views'], Tensor):
                    batch['num_views'] = unwrap(batch['num_views'])

                batch = self.__extend_with_batch_defaults(batch=batch)
            except StopIteration:
                self.reset_epoch()
                return None
            except Exception as ex:
                batch = None
                Console_UI().warn_user(f'Failed to load batch: "{ex}" for {self}')
                traceback.print_exception(type(ex), ex, ex.__traceback__)

            i += 1
            if i >= 5:
                raise RuntimeError(f'Failed multiple times when trying to retrieve batch for {self}')

        # To debug the images you can check output using debug_write_batch_encoded_images_and_quit()

        return batch

    def __get_iterator(self) -> Iterable[DataLoader]:
        if self.__dataloader_iterator is None:
            self.__no_iterators += 1
            loader = DataLoader(
                dataset=Batch_Loader(
                    dataset_name=self.__dataset_name,
                    experiment_name=self.name,
                    batch_size=int(math.ceil(self.__cfgs.batch_size * self.__batch_size_multiplier)),
                    view_dropout=self.__view_dropout,
                    annotations=self.__annotations,
                    explicit_modalities=self.__explicit_modalities,
                ),
                batch_size=1,  # The batch size is decided when generating the bins
                shuffle=False,  # The bin generation shuffles
                num_workers=Global_Cfgs().get('num_workers'),
                collate_fn=collate_factory(keys_2_ignore=('time')),
                pin_memory=gpu_activated,
                persistent_workers=False,  # At each iteration we want to recreate a new loader
                worker_init_fn=worker_init_fn(self.__no_iterators))
            self.__dataloader_iterator = iter(loader)
        return self.__dataloader_iterator

    def reset_epoch(self):
        self.__reset_iterator()

    def __extend_with_batch_defaults(self, batch: Dict[str, Any]):
        batch.update({
            'dataset_name': self.__dataset_name,
            'experiment_name': self.name,
            'results': defaultdict(dict),
            'loss': defaultdict(dict),
            'results_hooks': set(),
        })
        return batch

    def __setup_annotations(self):
        rel_path = self.__cfgs.annotations_path
        fm = File_Manager()
        self.__annotations = fm.read_csv_annotations(
            dataset_name=self.__dataset_name,
            annotations_rel_path=rel_path,
            multi_view_per_sample=self.__cfgs.multi_view_per_sample,
        )

        if self.__annotations is None:
            annotations_url = self.__cfgs.annotations_url
            available_csvs_str = '\', \''.join(fm.get_available_csvs(self.__dataset_name))
            Console_UI().inform_user(
                f'"{rel_path}" does not exist among the available datasets: "{available_csvs_str}".'
                '\nDownloading from:\n {annotations_url}')
            fm.download_zip_file(
                url=annotations_url,
                dataset_name=self.__dataset_name,
            )

            self.__annotations = fm.read_csv_annotations(
                dataset_name=self.__dataset_name,
                annotations_rel_path=rel_path,
                multi_view_per_sample=self.__multi_view_per_sample,
            )

        if Global_Cfgs().get('test_run'):
            self.__annotations = self.__annotations[self.__annotations.index.get_level_values(0) < 100]

    def end_epoch(self, summary: dict, scene_name: str):
        summary['dataset_name'] = self.__dataset_name
        summary['experiment_name'] = self.name
        summary['modalities'] = defaultdict(dict)
        raw_csv_data = []
        for _, modality in self.__modalities.items():
            if isinstance(modality, Base_Runtime_Value):
                raw_csv_data.extend(modality.get_runtime_values())
            modality.report_epoch_summary(summary)
            if modality.is_explicit_modality() and modality.is_csv():
                if isinstance(modality.content, pd.Series):
                    raw_csv_data.append(modality.content)
                elif isinstance(modality.content, pd.DataFrame):
                    [
                        raw_csv_data.append(modality.content[c])
                        for c in modality.content
                        # Avoid duplicated columns - this is usually die to that
                        # the original data hase several mappings, e.g. a regression
                        # for a point but that point could also be part of a line
                        if c not in [r.name for r in raw_csv_data]
                    ]
                else:
                    raise ValueError(f'The content type of {modality.get_name()} is not implemented')

        # We want the column estimates to be close to eachother in the final output
        raw_csv_data.sort(key=lambda v: v.name if hasattr(v, 'name') else -1)

        File_Manager().write_csv_annotation(
            annotations=pd.concat(raw_csv_data, axis=1),
            dataset_name=self.__dataset_name,
            experiment_file_name=f'{scene_name}_{self.__save_annotations_filename}',
        )
        return summary

    def __setup_modalities(self):
        modality_name = '_unitialized_'
        try:
            for modality_name, modality_cfgs_raw in self.__cfgs.modalities.items():
                modality_cfgs_dict = modality_cfgs_raw.dict()
                if (extra_settings := self.__cfgs.experiment_modality_settings.get(modality_name)) is not None:
                    modality_cfgs_dict.update({k: v for k, v in extra_settings.dict().items() if v is not None})

                self.__init_modality(modality_name, modality_cfgs_dict=modality_cfgs_dict)
        except pd.errors.EmptyDataError as error:
            path = self.__cfgs.annotations_path
            ds = self.__dataset_name
            msg = f'No data for {ds} in {path} when setting up modality \'{modality_name}\': {error}'
            raise pd.errors.EmptyDataError(msg)

    def __init_modality(self, modality_name: str, modality_cfgs_dict: Dict[str, Any]) -> None:
        modality_cfgs_dict['num_jitters'] = self.__cfgs.num_jitters
        modality_cfgs_dict['img_root'] = self.__cfgs.img_root
        modality_cfgs_dict['signal_to_noise_ratio'] = self.__cfgs.signal_to_noise_ratio
        modality_cfgs_dict['loss_weight_type'] = self.__cfgs.loss_weight_type
        modality_cfgs_dict['ignore_index'] = self.ignore_index

        try:
            modality_cfgs = Modality_Cfg_Parser.model_validate(modality_cfgs_dict).root
        except ValidationError as e:
            raise ValueError(f'Failed to handle {modality_cfgs_dict} for {modality_name}, got: {e}')

        modality_name = modality_name.lower()

        start_time = time.time()
        Modality, content, dictionary = get_modality_and_content(
            annotations=self.__annotations,
            name=modality_name,
            cfgs=modality_cfgs,
            ignore_index=self.ignore_index,
        )

        mc = self.__cfgs.min_channels
        if mc is None:
            mc = 2
        modality = Modality(dataset_name=self.__dataset_name,
                            modality_name=modality_name,
                            modality_cfgs=modality_cfgs,
                            content=content,
                            dictionary=dictionary,
                            spatial_transform=self.__cfgs.spatial_transform,
                            min_channels=mc)

        if modality.is_explicit_modality():
            self.__explicit_modalities[modality_name] = modality
            if modality.is_input_modality():
                self.__explicit_input_modalities[modality_name] = modality
            elif modality.is_output_modality():
                self.__explicit_output_modalities[modality_name] = modality
            else:
                raise KeyError('Explicit Modalities should either be input or output')
        elif modality.is_implicit_modality():
            self.__implicit_modalities[modality_name] = modality
        else:
            raise KeyError(f'Modality {modality_name} is neither implicit or explicit')

        # Add explicit and implicit modalities
        # Todo - Ali: why do we need to have this split? When do we have the case were a modality is neither
        self.__modalities.update(self.__explicit_modalities)
        self.__modalities.update(self.__implicit_modalities)

        if not Global_Cfgs().get('silent_init_info'):
            Console_UI().inform_user(
                info=f'Initializing {modality_name} modality in {self} in ' +
                f'{1000 *(time.time() - start_time):.0f} milliseconds',
                debug=(modality_cfgs_dict),
            )

    def get_modality(
        self,
        modality_name: str,
        modality_cfgs: Union[Modality_Cfg_Base, dict, None] = None,
    ) -> Any_Modality:
        modality_name = modality_name.lower()
        if modality_name not in self.__modalities:
            if isinstance(modality_cfgs, Modality_Cfg_Base):
                modality_cfgs = modality_cfgs.dict()
            self.__init_modality(modality_name, modality_cfgs_dict=modality_cfgs)

        modality = self.__modalities[modality_name]
        if modality_cfgs is not None:
            modality.update_data_dimensions(modality_cfgs)
        return modality

    def get_modalities(self) -> List[str]:
        modalities = self.__cfgs.modalities
        return [self.get_modality(name.lower(), cfgs) for name, cfgs in modalities.items()]

    def get_explicit_classification_modality_names(self) -> List[str]:
        return [
            m.lower()
            for m in self.get_explicit_modality_names()
            if self.get_modality(m).is_output_modality() and self.get_modality(m).is_classification()
        ]

    def get_explicit_regression_modality_names(self) -> List[str]:
        return [
            m.lower()
            for m in self.get_explicit_modality_names()
            if self.get_modality(m).is_output_modality() and self.get_modality(m).is_regression()
        ]

    def get_explicit_input_modality_names(self) -> List[str]:
        return [m.lower() for m in self.get_explicit_modality_names() if self.get_modality(m).is_input_modality()]

    def get_explicit_modality_names(self) -> List[str]:
        return [m.lower() for m, _ in self.__cfgs.modalities.items()]

    def drop_modality(self, name: str) -> bool:
        """Remove a modality, e.g. a Style if the graph indicates that it
        is not being used.

        Args:
            name (str): The name of the modality

        Returns: True if existed and deleted
        """
        exists = False
        if name in self.__explicit_modalities:
            exists = True
            del self.__explicit_modalities[name]

        if name in self.__explicit_input_modalities:
            exists = True
            del self.__explicit_input_modalities[name]

        if name in self.__explicit_output_modalities:
            exists = True
            del self.__explicit_output_modalities[name]

        if name in self.__modalities:
            exists = True
            del self.__modalities[name]

        return exists

    def get_implicit_classification_modality_names(self):
        return [
            self.get_modality(m.lower()).get_implicit_modality_name()
            for m in self.get_explicit_classification_modality_names()
        ]

    def get_implicit_regression_modality_names(self):
        return [
            self.get_modality(m.lower()).get_implicit_modality_name()
            for m in self.get_explicit_regression_modality_names()
        ]

    def get_implicit_input_modality_names(self):
        return [
            self.get_modality(m.lower()).get_implicit_modality_name() for m in self.get_explicit_input_modality_names()
        ]

    def get_implicit_modality_names(self):
        return [self.get_modality(m.lower()).get_implicit_modality_name() for m in self.get_explicit_modality_names()]

    def get_explicit_pseudo_output_modality_names(self) -> List[str]:
        return [
            f'pseudo_{m.lower()}' for m in self.get_explicit_modality_names()
            if self.get_modality(m).is_output_modality() and self.get_modality(m).has_pseudo_label()
        ]

    def get_implicit_pseudo_output_modality_names(self):
        return [f'implicit_{m}' for m in self.get_explicit_pseudo_output_modality_names()]

    def get_modality_cfgs(self, modality_name: str) -> Dict[str, any]:
        modality_name = modality_name.lower()
        # check if modality is already created
        if modality_name.lower() in self.__modalities:
            return self.__modalities[modality_name.lower()].get_modality_cfgs()
        # check if it's an implicit modality
        # check if modality is pseudo modality
        elif (modality_name.startswith('pseudo_') and modality_name[len('pseudo_'):].lower() in self.__modalities):
            modality = self.__modalities[modality_name[len('pseudo_'):].lower()]
            original_modality_cfgs = modality.get_modality_cfgs()
            modality_cfgs = {
                'type': 'pseudo_label',
                'consistency': original_modality_cfgs['consistency'],
                'num_channels': modality.get_num_classes(),
                'tensor_shape': list(original_modality_cfgs['tensor_shape']),
            }
            modality = self.get_modality(modality_name, modality_cfgs)
            return modality.get_modality_cfgs()

        for _, modality in self.__modalities.items():
            if (modality.is_explicit_modality()
                    and modality.get_implicit_modality_name().lower() == modality_name.lower()):
                return modality.get_implicit_modality_cfgs()
            # check if it's an implicit pseudo modality

        raise KeyError(f'Unknown modality "{modality_name}" in dataset "{self}"')

    def get_model_cfgs(self, modality_name: str) -> Optional[Dict[str, Any]]:
        modality_name = modality_name.lower()
        if modality_name in self.__explicit_modalities:
            modality = self.__explicit_modalities[modality_name.lower()]
            if modality.is_explicit_modality():
                return self.__explicit_modalities[modality_name.lower()].get_model_cfgs()
        if modality_name.startswith('pseudo_'):
            modality_name = modality_name[len('pseudo_'):]
            modality = self.__explicit_modalities[modality_name.lower()]
            if modality.is_explicit_modality():
                modality_cfgs = self.__explicit_modalities[modality_name.lower()].get_model_cfgs().copy()
                modality_cfgs['heads'] = [f'pseudo_{s}' for s in modality_cfgs['heads']]
                modality_cfgs['tails'] = [f'pseudo_{s}' for s in modality_cfgs['tails']]
                return modality_cfgs
        return None

    def get_model_name(self, modality_name: str) -> str:
        if modality_name in self.__explicit_modalities:
            model_name = self.__modalities[modality_name.lower()].get_model_name()
            return model_name
        if modality_name.startswith('pseudo_'):
            modality_name = modality_name[len('pseudo_'):]
            model_name = 'pseudo_%s' % self.__modalities[modality_name.lower()].get_model_name()
            return model_name
        raise KeyError('Unsupported modality %s' % modality_name)

    def get_loss_name(self, modality_name: str) -> str:
        return self.__modalities[modality_name.lower()].get_loss_name()

    def get_implicit_modality_name(self, explicit_modality_name):
        explicit_modality_name = explicit_modality_name.lower()
        if explicit_modality_name in self.__modalities:
            modality = self.__modalities[explicit_modality_name]
            if modality.is_explicit_modality():
                return modality.get_implicit_modality_name()
            else:
                modality_name = explicit_modality_name[len('pseudo_'):]
                if modality_name in self.__modalities:
                    return f'pseudo_{self.__modalities[modality_name].get_implicit_modality_name()}'

    def get_name(self) -> str:
        return self.name
