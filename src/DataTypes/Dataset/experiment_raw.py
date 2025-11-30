from typing import Dict, List, Literal, Optional, Union

from pydantic import Field

from DataTypes.enums import SpatialTransform
from DataTypes.Modality.cfg_groups import Any_CSV_Modality_Cfgs, Any_Modality_Cfg
from DataTypes.general import BaseModelWithGet


class Dataset_Experiment_Cfg_Raw(BaseModelWithGet):
    annotations_path: str = ...
    "The csv name that we are using"

    num_jitters: int
    "Not used?"

    spatial_transform: SpatialTransform = ...
    "How the jitters should transform the image"

    modalities: Union[Literal['same_as_train_set'], Dict[str, Any_CSV_Modality_Cfgs]]
    "Key outcomes that we are interested in e.g. BP_ankle"

    shared_modalities: Optional[List[str]] = Field(default=None)
    "Modalities defined in the share directory"


class Dataset_Experiment_Cfg_Fixed(Dataset_Experiment_Cfg_Raw):
    modalities: Dict[str, Any_Modality_Cfg]
    "Key outcomes that we are interested in e.g. BP_ankle"

    shared_modalities: Optional[List[str]] = Field(default=None)
    "Modalities defined in the share directory"
