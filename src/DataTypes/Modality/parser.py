from pydantic import RootModel

from .cfg_groups import Any_Modality_Cfg


class Modality_Cfg_Parser(RootModel[Any_Modality_Cfg]):
    pass
