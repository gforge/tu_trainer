from typing import Optional

from pydantic import Field

from DataTypes.enums import TaskType
from .activators import Task_Cfg_Activators
from .inheritance import Task_Cfg_Inheritance


class Task_Cfg_Raw(Task_Cfg_Inheritance):
    task_type: TaskType = ...
    "Type of task"

    dataset_name: str = ...
    "The name of the dataset to use for this task"

    pseudo_set_name: Optional[str] = Field(default=None)
    """The name of the pseudo-label dataset to use

    When augmenting the data with pseudo labels that a supplementary
    network uses we need to supply additional variables
    """

    train_set_name: str = ...
    "The main training dataset"

    val_set_name: str = ...
    "The validation set name"

    test_set_name: Optional[str] = Field(default=None)
    "The test set name"

    validate_when_epoch_is_devisable_by: int = 5
    "How often validation should occur between epochs"

    batch_size_multiplier: float = 1
    "Some tasks require more memory and we need to reduce the memory footprint, value 0 < x <= 1, defaults to 1"

    apply: Task_Cfg_Activators = ...
    "Features that we want to apply to this Task"
