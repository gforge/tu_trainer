from typing import Dict, Optional

from pydantic import Field
from pydantic.types import PositiveInt

from .experiment_raw import Dataset_Experiment_Cfg_Raw
from DataTypes.general import BaseModelWithGet, Proportion


class Dataset_Cfg_Raw(BaseModelWithGet):
    img_root: str = ...
    "Where the images are situated, usually 'xray_root' for xray images"

    annotations_url: Optional[str] = Field(default=None)
    "If we want to download the annotations from a web-site then set this to the http"

    signal_to_noise_ratio: Proportion
    "How correct the labels are estimated, 0.95 are highly correct labels while 0.7 are noisy labels"

    multi_view_per_sample: bool = ...
    "If there are multiple views per sample/exam"

    to_each_view_its_own_label: bool = ...
    "If eah label has it's own label"

    experiments: Dict[str, Dataset_Experiment_Cfg_Raw]
    "The experiments that we are running, should usually be train_set, val_set, & test_set"

    min_channels: Optional[PositiveInt] = Field(default=None)
    "The minimum number of channels to use for the implicit modalities"

    ignore_index: int = -100
    "The index to ignore, i.e. missing value"
