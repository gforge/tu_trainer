import numpy as np
from typing import Tuple, Callable
from dataclasses import dataclass


@dataclass
class _ImageType():
    color: bool
    cv2_format: bool
    three_dim: bool


def _get_image_type(im: np.ndarray) -> _ImageType:
    assert im.ndim == 2 or im.ndim == 3

    if im.ndim == 2:
        return _ImageType(color=False, three_dim=False, cv2_format=False)

    im_type = _ImageType(color=True, three_dim=True, cv2_format=True)
    if im.shape[0] == 3 or im.shape[0] == 1:
        im_type.cv2_format = False

    if im.shape[0] == 1 or im.shape[2] == 1:
        im_type.color = False

    return im_type


def convert_image_into_cv2_format(im: np.ndarray) -> Tuple[np.ndarray, _ImageType]:
    im_type = _get_image_type(im=im)
    if not im_type.three_dim:
        im = add_third_dimension_if_needed(im, along_axis=2)

    if not im_type.cv2_format and im_type.three_dim:
        # The copy is required or opencv fails to identify that it has the correct structure
        im = im.transpose((1, 2, 0)).copy()

    if not im_type.color:
        # For this use case we do not want to convert using the color correct conversion
        # as transforming back can be tricky, i.e. visual_appeal = True
        im = grayscale_2_color(im=im, along_axis=2, visual_appeal=False)

    return im, im_type


def revert_cv2_image_to_original_format(im: np.ndarray, org_image_type: _ImageType) -> np.ndarray:
    color_axis = 2
    if not org_image_type.cv2_format:
        im = im.transpose((2, 0, 1))
        color_axis = 0

    if org_image_type.color:
        if im.shape[0] == 1:
            im = grayscale_2_color(im=im, visual_appeal=False, along_axis=color_axis)
    else:
        if im.shape[color_axis] == 3:
            if color_axis == 0:
                im = im[[0], :, :]
            else:
                im = im[:, :, [0]]

    if not org_image_type.three_dim:
        if color_axis == 0:
            im = im[0, :, :]
        else:
            im = im[:, :, 0]

    return im


def convert_image_to_cv2_and_back_to_original_format(cv2_draw_fn: Callable, **kwargs):
    """
    As the draw function is a OpenCV function i expects the
    third shape to be color while the network has that as the first.
    This decorator handles the conversion transparently and returns
    in the same format as received.
    """

    def inner(im, **kwargs):
        im, org_image_type = convert_image_into_cv2_format(im)

        ret = cv2_draw_fn(im=im.copy(), **kwargs)

        if type(ret) is np.ndarray:
            return revert_cv2_image_to_original_format(im=ret, org_image_type=org_image_type)

        if type(ret) is tuple and len(ret) == 2:
            return revert_cv2_image_to_original_format(im=ret[0], org_image_type=org_image_type), ret[1]

        raise ValueError(f'Could not handle decorated {cv2_draw_fn.__name__}() returned format: {type(ret)}')

    return inner


def add_third_dimension_if_needed(im: np.ndarray, along_axis: int) -> np.ndarray:
    if im.ndim == 3:
        return im

    if im.ndim == 2:
        if along_axis == 0:
            return im.reshape([1, *im.shape])

        if along_axis == 2:
            return im.reshape([*im.shape, 1])

        raise IndexError(f'Can only add dimensions either first or last, got {along_axis}')

    raise ValueError(f'Expected 2 or 3-dim image input, got {im.ndim} dimensions')


def grayscale_2_color(im: np.ndarray, visual_appeal: bool, along_axis: int) -> np.ndarray:
    im = add_third_dimension_if_needed(im, along_axis=along_axis)
    im = im.repeat(3, axis=along_axis)

    if visual_appeal:
        if along_axis == 2:
            im = 0.2989 * im[:, :, 0] + 0.5870 * im[:, :, 1] + 0.1140 * im[:, :, 2]
        elif along_axis == 0:
            im = 0.2989 * im[0, :, :] + 0.5870 * im[1, :, :] + 0.1140 * im[2, :, :]
        else:
            raise IndexError(f'Invalid axis: {along_axis}, expected first or last axis')

    return im
