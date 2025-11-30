from typing import Optional

from DataTypes.general import Proportion
from DataTypes.Task.inheritance import Task_Cfg_Inheritance


class Scene_Cfg_Inheritance(Task_Cfg_Inheritance):
    """Values that can be inherited from top level
    """
    forward_noise: Optional[Proportion] = None
    "The white noise between each network to regularize with, can also be set at a scenario level"
