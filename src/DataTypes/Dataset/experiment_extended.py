from typing import Any, Dict, Optional

from pydantic import PositiveInt, Field

from DataTypes.Modality.cfg_groups import Any_Modality_Cfg
from .experiment_raw import Dataset_Experiment_Cfg_Raw
from .extended import Dataset_Cfg_Extended
from .experiment_inheritance import Dataset_Experiment_Cfg_Inherits


class Dataset_Experiment_Cfg_Extended(Dataset_Experiment_Cfg_Raw, Dataset_Experiment_Cfg_Inherits):
    name: str = ...
    "The name of the experiment"

    min_channels: PositiveInt = Field(default=None)
    "The minimum number of channels to use for the implicit modalities"

    dataset_name: str = ...
    "The name of the dataset"

    multi_view_per_sample: bool = Field(default=None)
    "Has the dataset multiple cases per sample or is each row unique"

    img_root: str = Field(default=None)
    "The image root"

    signal_to_noise_ratio: float = Field(default=None)
    "The signal to noise ratio, i.e. how correct the labels are"

    modalities: Dict[str, Any_Modality_Cfg]
    "Key outcomes that we are interested in e.g. BP_ankle"

    annotations_url: Optional[str] = Field(default=None)
    "If we want to download the annotations from a web-site then set this to the http"

    ignore_index: int = Field(default=None)
    "The index to ignore, i.e. missing value"

    def __init__(self, **data: Any) -> None:
        dataset_cfgs: Dataset_Cfg_Extended = data['dataset_cfgs']
        del data['dataset_cfgs']
        data['dataset_name'] = dataset_cfgs.name
        # Default values are not properly filled if none is supplied here
        data = {k: v for k, v in data.items() if v is not None}

        super().__init__(**data)

        self._extend_with_another_models_data(dataset_cfgs)

        if dataset_cfgs.experiment_modality_settings is not None:
            # Due to the default dict() we need to force this copy
            self.experiment_modality_settings = dataset_cfgs.experiment_modality_settings
