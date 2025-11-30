import re
from typing import Any, Dict, Optional
from pydantic import Field

from pydantic.types import PositiveInt

from DataTypes.Task.raw import Task_Cfg_Raw
from DataTypes.scenario_cfg import Scenario_Cfg
from .raw import Scene_Cfg_Raw


class Scene_Cfg_Extended(Scene_Cfg_Raw):
    tasks: Dict[str, Task_Cfg_Raw] = ...
    "The names of the tasks that we want to train in these scene"

    scenario_name: str
    "The name of the scenario"

    min_channels: Optional[PositiveInt] = Field(default=None)
    "The minimum number of channels to use for the implicit modalities"

    main_task: Optional[str] = Field(default=None)
    "The main task. If none is defined then it defaults to the first scene"

    batch_size: Optional[int] = Field(default=None)
    "The batch size"

    backprop_every: Optional[PositiveInt] = Field(default=None)
    """When to backprop in training

    If we have small batches due to memory limitation we may want to run backprop less than every iteration
    so that we in practice get larger batch size and a more stable training."""

    def __init__(self, **data: Any) -> None:
        scenario_cfgs: Scenario_Cfg = data['scenario_cfgs']
        del data['scenario_cfgs']
        data['scenario_name'] = scenario_cfgs.name

        super().__init__(**data)

        self._extend_with_another_models_data(scenario_cfgs)
        self.__resolve_main_task()

    def __resolve_main_task(self):
        from UIs.console_UI import Console_UI

        default_task = list(self.tasks.keys())[0]
        if self.main_task is None:
            self.main_task = default_task
            Console_UI().inform_user(f'Picked the {self.main_task} as the main task to define general epoch')
        elif self.main_task not in self.tasks:
            regex_main_matches = list(filter(re.compile(self.main_task).search, self.tasks.keys()))
            if (len(regex_main_matches) > 0):
                self.main_task = regex_main_matches[0]
                Console_UI().inform_user(f'Picked the {self.main_task} as the main task to define general epoch')
            else:
                Console_UI().inform_user(f'The {self.main_task} couldn\'t be resolved,' +
                                         ' picking {default_task} as the main task to define general epoch')
                self.main_task = default_task
