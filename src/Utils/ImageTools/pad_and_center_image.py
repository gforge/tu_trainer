import numpy as np
import cv2
from typing import Tuple, TypeVar, Generic
from .color_helpers import convert_image_to_cv2_and_back_to_original_format
from GeneralHelpers import init_nan_array, extract_and_set_shape, append_shape, unwrap
from .constants import backend

IndexeableShape = TypeVar('IndexeableShape', Tuple[int, int], np.ndarray)
Number = TypeVar('Number', int, float)


class _Padding(Generic[Number]):

    # Follows the CSS-style order
    order = ('top', 'right', 'bottom', 'left')

    def __init__(self, left: Number, right: Number, top: Number, bottom: Number):
        super().__init__()
        self.data = {'left': left, 'right': right, 'top': top, 'bottom': bottom}

    def __repr__(self):
        return f'<Padding left: {self.left}, right: {self.right}, top: {self.top}, bottom: {self.bottom}>'

    def set_value(self, name: str, value: Number):
        if name in self.data:
            self.data[name] = value
            return self

        raise KeyError(f'The key {name} does not exist')

    def get_value(self, name: str):
        if name in self.data:
            return self.data[name]

        raise KeyError(f'The key {name} does not exist')

    def to_numpy(self) -> np.array:
        return np.array([self.get_value(name=element) for element in _Padding.order], dtype=np.float64)

    @staticmethod
    def from_pickled(array):
        array = unwrap(array)
        assert len(array.shape) == 1
        assert array.shape[0] == 4
        args = {element: array[idx] for idx, element in enumerate(_Padding.order)}
        return _Padding(**args)

    @property
    def left(self):
        return self.get_value('left')

    @property
    def right(self):
        return self.get_value('right')

    @property
    def top(self):
        return self.get_value('top')

    @property
    def bottom(self):
        return self.get_value('bottom')

    @property
    def available_width(self) -> Number:
        return 1. - (self.left + self.right)

    @property
    def available_height(self) -> Number:
        return 1. - (self.top + self.bottom)


class PaddingDetails():

    def __init__(
        self,
        pixels: _Padding,
        proportions: _Padding,
        org_shape: IndexeableShape,
        padded_shape: IndexeableShape,
    ):
        super().__init__()
        self.pixels = pixels
        self.proportions = proportions
        self.org_shape = org_shape
        self.padded_shape = padded_shape

    def to_numpy(self) -> np.ndarray:
        base = init_nan_array(shape=(4, 4), dtype=np.float64)
        base[0] = self.pixels.to_numpy()
        base[1] = self.proportions.to_numpy()

        append_shape(destination=base, shape=self.org_shape, main_idx=2)
        append_shape(destination=base, shape=self.padded_shape, main_idx=3)
        return base

    @staticmethod
    def from_pickled(array):
        array = unwrap(array)
        assert len(array.shape) == 2
        assert array.shape[0] == 4
        assert array.shape[1] == 4

        args = {'pixels': array[0], 'proportions': array[1]}
        args = {key: _Padding.from_pickled(value) for key, value in args.items()}

        extract_and_set_shape(destination=args, source_array=array, name='org_shape', main_idx=2)
        extract_and_set_shape(destination=args, source_array=array, name='padded_shape', main_idx=3)

        return PaddingDetails(**args)

    def transform_coordinates_2_padded_space(self, coordinates: np.ndarray, inplace: bool = True):
        if not inplace:
            coordinates = coordinates.copy()

        coordinates[:, 0] = coordinates[:, 0] * self.proportions.available_width + self.proportions.left
        coordinates[:, 1] = coordinates[:, 1] * self.proportions.available_height + self.proportions.top
        return coordinates

    def transform_coordinates_from_padded_space(self, coordinates: np.ndarray, inplace: bool = True):
        if not inplace:
            coordinates = coordinates.copy()

        coordinates[:, 0] = (coordinates[:, 0] - self.proportions.left) / self.proportions.available_width
        coordinates[:, 1] = (coordinates[:, 1] - self.proportions.top) / self.proportions.available_height
        return coordinates


def numpy_center_copy_image(im2paste: np.ndarray, max_shape: int) -> np.ndarray:
    larger_img = np.empty((max_shape, max_shape, im2paste.shape[2]), dtype=im2paste.dtype)
    larger_img.fill(0)
    diff = np.floor((np.array(larger_img.shape) - np.array(im2paste.shape)) / 2).astype(int)
    larger_img[diff[0]:(diff[0] + im2paste.shape[0]), diff[1]:(diff[1] + im2paste.shape[1])] = im2paste
    return larger_img


@convert_image_to_cv2_and_back_to_original_format
def pad_and_center_image(im: np.ndarray, skip_pad: bool = False) -> Tuple[np.ndarray, PaddingDetails]:
    shape = im.shape
    assert len(shape) == 2 or (len(shape) == 3 and shape[2] == 3), \
        'Expected grayscale or cv2 formatted image with color at third position'

    padding = _Padding(left=0, right=0, top=0, bottom=0)
    proportion = _Padding(left=0, right=0, top=0, bottom=0)

    if skip_pad:
        return im, PaddingDetails(pixels=padding, proportions=proportion, org_shape=shape, padded_shape=im.shape)

    max_idx: int = np.argmax(shape)
    max_shape: int = shape[max_idx]
    assert max_shape < 1e5, f'Huge imax siza!? {max_shape} picked from idx {max_idx} from image shape: {shape}'

    portrait: bool = False
    if max_idx == 0:
        portrait = True

    def _adapt_args(border_name_1, border_name_2):
        extra = max_shape - np.min(shape[0:2])
        pad_width = int(extra / 2)
        bonus = (0, int(extra % 2))
        for i, bn in enumerate([border_name_1, border_name_2]):
            padding.set_value(name=bn, value=pad_width + bonus[i])
            padding.set_value(name=bn, value=pad_width + bonus[i])
            proportion.set_value(name=bn, value=padding.get_value(name=bn) / max_shape)

    if portrait:
        _adapt_args(border_name_1='left', border_name_2='right')
    else:
        _adapt_args(border_name_1='top', border_name_2='bottom')

    if backend == 'skimage':
        out_img = numpy_center_copy_image(im2paste=im, max_shape=max_shape)
    else:
        border_arguments = {'src': im, 'borderType': cv2.BORDER_CONSTANT, 'value': 0, **padding.data}
        out_img = cv2.copyMakeBorder(**border_arguments)

    return out_img, PaddingDetails(pixels=padding, proportions=proportion, org_shape=shape, padded_shape=out_img.shape)
