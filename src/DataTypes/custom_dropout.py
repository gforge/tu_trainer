from pydantic import BaseModel


class Custom_Dropout_Cfg(BaseModel):
    width: int = ...
    "The proportion of the image width to use"

    height: int = ...
    "The proportion of the image height to use"

    number: int = ...
    "The number of blocks to erase"

    prop_border_to_ignore: float = ...
    """The proportion of the border to ignore from dropping

    We often don't want to erase pieces at the border and thus
    it may be useful to drop cases at the borders.
    """
