from typing import Any, Dict, Optional
from pydantic import Field
from pydantic.types import PositiveInt

from DataTypes.Dataset.experiment_raw import Dataset_Experiment_Cfg_Fixed
from DataTypes.Task.extended import Task_Cfg_Extended

from .raw import Dataset_Cfg_Raw
from .experiment_inheritance import Dataset_Experiment_Cfg_Inherits


class Dataset_Cfg_Extended(Dataset_Cfg_Raw, Dataset_Experiment_Cfg_Inherits):
    name: str
    "The name of the dataset"

    task_name: str
    "The name of the task that this ds belongs to"

    scene_name: str
    "The scene name of the task that this ds belongs to"

    min_channels: Optional[PositiveInt] = Field(default=None)
    "The minimum number of channels to use for the implicit modalities"

    experiments: Dict[str, Dataset_Experiment_Cfg_Fixed]
    "The experiments that we are running, should usually be train_set, val_set, & test_set"

    def __init__(self, **data: Any) -> None:
        task_cfgs: Task_Cfg_Extended = data['task_cfgs']
        del data['task_cfgs']
        data['task_name'] = task_cfgs.name
        data['scene_name'] = task_cfgs.scene_name

        super().__init__(**data)

        self._extend_with_another_models_data(task_cfgs)

        # Due to the default dict() we need to force this copy
        self.experiment_modality_settings = task_cfgs.experiment_modality_settings
