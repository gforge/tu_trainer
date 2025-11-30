from __future__ import annotations
from DataTypes.Dataset.experiment_extended import Dataset_Experiment_Cfg_Extended
from DataTypes.Dataset.extended import Dataset_Cfg_Extended
from DataTypes.enums import LossWeightType
from .experiment_set import Experiment_Set
from typing import List, Dict, Optional


class CSV_Dataset:

    def __init__(self, cfgs: Dataset_Cfg_Extended):
        self.__cfgs = cfgs
        self.__setup_experiments()

    @property
    def dataset_name(self) -> str:
        return self.__cfgs.name

    def __setup_experiments(self):
        self.experiments: Dict[str, Experiment_Set] = {}
        # Sometimes we have the same filename and then we want to avoid that the save occurrs to the
        # same name that we used for saving
        used_save_names: List[str] = []
        for experiment_name, experiment_cfgs in self.__cfgs.experiments.items():

            cfgs_dict = experiment_cfgs.dict()
            cfgs_dict.update(
                dataset_cfgs=self.__cfgs,
                min_channels=self.__cfgs.min_channels,
                view_dropout=self.__cfgs.view_dropout,
                experiment_modality_settings=self.__cfgs.experiment_modality_settings,
                loss_weight_type=self.__cfgs.loss_weight_type,
                scene_name=self.__cfgs.scene_name,
                task_name=self.__cfgs.task_name,
                name=experiment_name,
            )
            cfgs_dict = {k: v for k, v in cfgs_dict.items() if v is not None}
            cfgs = Dataset_Experiment_Cfg_Extended(**cfgs_dict)

            save_annotations_filename: str = experiment_cfgs.annotations_path
            if save_annotations_filename in used_save_names:
                save_annotations_filename = f'{experiment_name}_{save_annotations_filename}'
            used_save_names.append(save_annotations_filename)

            self.experiments[experiment_name] = Experiment_Set(cfgs=cfgs,
                                                               save_annotations_filename=save_annotations_filename)

    def set_batch_size_multiplier(self, number: int) -> CSV_Dataset:
        """
        Updates the batch multiplier at runtime for each of the experiments.

        This allows us to reduce the size when we are doin memory heavy scenes that
        for instance involve autoencoders.
        """
        for name in self.experiments.keys():
            self.experiments[name].set_batch_size_multiplier(number)

        return self

    def set_view_dropout(self, view_dropout: Optional[float]) -> CSV_Dataset:
        """
        Updates the view dropout at runtime for each of the experiments.

        The dropout controlles how many images will be used at each step
        from the total. 0 or None indicates that all will be used.
        """
        for name in self.experiments.keys():
            self.experiments[name].set_view_dropout(view_dropout)

        return self

    def set_loss_weight_type(self, loss_weight_type: LossWeightType) -> CSV_Dataset:
        """
        Updates the loss weight type at runtime for each of the experiments.
        """
        for name in self.experiments.keys():
            self.experiments[name].set_loss_weight_type(loss_weight_type)

        return self

    def get_name(self) -> str:
        return self.dataset_name

    def get_output_modality_names(self, experiment_name: str) -> List[str]:
        return self.experiments[experiment_name].get_output_modality_names()

    def get_input_modality_names(self, experiment_name):
        return self.experiments[experiment_name].get_input_modality_names()
