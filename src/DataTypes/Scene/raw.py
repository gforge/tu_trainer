from typing import List, Any, Dict, Optional
from pydantic import Field, PositiveInt

from DataTypes.general import Proportion
from DataTypes.learning_rate_cfg import Learning_Rate_Cfg
from .inheritance import Scene_Cfg_Inheritance


class Scene_Cfg_Raw(Scene_Cfg_Inheritance):
    name: str = ...
    "An identifier for the given scene"

    repeat: PositiveInt = 1
    "The number of times we want to repeat the entire scene (total epochs = repeat * epochs)"

    epochs: PositiveInt = ...
    "The number of runs through the main task"

    tasks: List[str] = Field(min_length=1)
    "The names of the tasks that we want to train in these scene"

    learning_rate: Learning_Rate_Cfg = ...
    "The learning rate settings"

    task_defaults: Dict[str, Any] = {}
    "Values that can are used by all tasks"

    stochastic_weight_averaging: bool = False
    "Use stochastic weighted averaging (SWA)"

    stochastic_weight_averaging_last: Optional[bool] = Field(default=False)
    "Use stochastic weighted averaging at the last repeat within a scene"

    has_pseudo_labels: Optional[bool] = Field(default=False)
    "Should pseudo labels be activated"

    pseudo_loss_factor: Optional[float] = Field(default=0.1)
    "How much pseudo loss from fictive labels compares to regular loss from manually annotated images"

    forward_noise: Proportion = 0.
    "The white noise between each network to regularize with, can also be set at a scenario level"

    use_custom_dropout: bool = False
    "Use custom block dropout"

    def __len__(self):
        return len(self.tasks)
