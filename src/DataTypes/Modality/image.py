from typing import Any, Literal, Optional
from pydantic import PositiveInt, Field

from DataTypes.enums import Colorspace, Consistency
from .other import Modality_Cfg_Base


class Modality_Image_Cfg(Modality_Cfg_Base):
    type: Literal['Image_from_Filename']

    consistency: Consistency = Consistency.d2
    "The consistency of an image should always be 2-dimensional"

    column_name: str
    "The column name with the text"

    colorspace: Colorspace
    "The allowed color space for the input images"

    scale_to: PositiveInt
    """The scale images to resize to

    If width and height are omitted the image input will be
    squared. Otherwise this should be one of the dimensions of
    either height or width.
    """

    width: PositiveInt = Field(default=None)
    "Resize images to this width before feeding to the network"

    height: PositiveInt = Field(default=None)
    "Resize images to this height before feeding to the network"

    keep_aspect: bool
    "Should we stretch the image to span the entire image space"

    modality: Literal['input']

    skip_padding: bool = False
    "Should we skip padding the image?"

    img_root: Optional[str] = Field(default=None)
    "The image root"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if all([dim is None for dim in (self.width, self.height)]):
            self.width = self.scale_to
            self.height = self.scale_to

        assert all([dim is not None for dim in (self.width, self.height)])
        assert any([dim == self.scale_to for dim in (self.width, self.height)]), \
            f'Scale to assumes that at least one of the width ({self.width})' + \
            f' or {self.height} match {self.scale_to}'
