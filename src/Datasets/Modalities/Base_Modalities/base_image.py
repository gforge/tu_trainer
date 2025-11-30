from abc import ABCMeta
from DataTypes import Colorspace, Modality_Image_Cfg

from .base_plane import Base_Plane


class Base_Image(Base_Plane[Modality_Image_Cfg, int], metaclass=ABCMeta):
    """
    We refer to any modality with 2D consistency as image, although it doesn't
    necessarily have to be an image.
    This class implements basic transformations that can be applied to 2D surfaces
    this include jittering, augmentation, etc
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_size = (self.height, self.width)

        self.__init_channels_from_colorspace()

    @property
    def scale_to(self) -> int:
        "The size that we want to rescale images to. If width/height are omitted this will replace them"
        return self._cfgs.scale_to

    @property
    def keep_aspect(self) -> bool:
        return self._cfgs.keep_aspect

    @property
    def colorspace(self) -> Colorspace:
        return self._cfgs.colorspace

    def __init_channels_from_colorspace(self):
        if self.colorspace == Colorspace.gray:
            self.set_channels(1)
        elif self.colorspace == Colorspace.rgb:
            self.set_channels(3)
        else:
            raise KeyError(f'Unknown colorspace {self.colorspace}. Colorspace can only be "Gray" or "RGB"')
