from typing import Dict, Optional
from pydantic import Field

from DataTypes.Modality.cfg_groups import Any_Modality_Cfg
from DataTypes.Modality.inheritance import Dataset_Modality_Cfg_Inheritance


class Dataset_Experiment_Cfg_Inherits(Dataset_Modality_Cfg_Inheritance):
    view_dropout: Optional[float] = Field(ge=0, lt=1, default=None)
    "Set if we want to drop sections of the view"

    batch_size_multiplier: float = Field(default=None)
    "Some tasks require more memory and we need to reduce the memory footprint, value 0 < x <= 1, defaults to 1"

    scenario_name: str = Field(default=None)
    "The name of the scenario"

    batch_size: int = Field(default=None)
    "The batch size"

    experiment_modality_settings: Dict[str, Any_Modality_Cfg] = Field(default_factory=dict)
    "Settings for specific modalities, e.g. custom image size"
