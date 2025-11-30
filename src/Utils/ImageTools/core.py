import numpy as np
from typing import Tuple, Callable, Union, Literal
import cv2
from skimage import io
from skimage.transform import warp, AffineTransform, resize

from DataTypes.enums import Colorspace

from .pad_and_center_image import pad_and_center_image
from .SpatialTransform import SpatialTransform
from .transformation_matrix import get_fixed_spatial_transform_matrix, get_random_spatial_transform_matrix
from .color_helpers import add_third_dimension_if_needed
from .plots import convert_image_to_cv2_and_back_to_original_format
from .constants import backend


@convert_image_to_cv2_and_back_to_original_format
def _transform_image(
    im: np.ndarray,
    output_size: Union[Tuple[int, int], np.array],
    augmentation_index: int,
    skip_pad: bool,
    skip_resize: bool,
    spatial_tranform_fn: Callable[[np.ndarray, Tuple[int, int], int], Tuple[np.ndarray, SpatialTransform]],
):
    im, pad_details = pad_and_center_image(im, skip_pad=skip_pad)
    # During loading this is actaully usually performed by calling fix_dimension_and_normalize()
    if not skip_resize:
        im = _resive_img(im, output_size=output_size, keep_aspect=True)

    M = spatial_tranform_fn(im.shape, output_size, augmentation_index)
    if backend == 'cv2':
        tr_im = cv2.warpPerspective(im, M, (int(output_size[1]), int(output_size[0])))
    elif backend == 'skimage':
        tform = AffineTransform(M)
        tr_im = warp(im, tform.inverse, output_shape=(int(output_size[0]), int(output_size[1])))
    else:
        raise KeyError(f'Invalid backend: {backend}')

    return tr_im, SpatialTransform(M=M, cropped_im_shape=tr_im.shape, pre_transform_shape=im.shape, padding=pad_details)


def get_random_transformed_image(
    im: np.array,
    output_size: np.array,
    augmentation_index: int,
    skip_pad: bool = False,
    skip_resize: bool = False,
):
    return _transform_image(
        im=im,
        output_size=output_size,
        augmentation_index=augmentation_index,
        skip_pad=skip_pad,
        skip_resize=skip_resize,
        spatial_tranform_fn=get_random_spatial_transform_matrix,
    )


def get_fixed_transformed_image(
    im: np.array,
    output_size: np.array,
    augmentation_index: int,
    skip_pad: bool = False,
    skip_resize: bool = False,
):
    return _transform_image(
        im=im,
        output_size=output_size,
        augmentation_index=augmentation_index,
        skip_pad=skip_pad,
        skip_resize=skip_resize,
        spatial_tranform_fn=get_fixed_spatial_transform_matrix,
    )


def fix_dimension_and_normalize(
    im: np.array,
    keep_aspect: bool,
    scale_to: int,
    colorspace: str,
) -> Tuple[np.ndarray, np.ndarray]:
    original_im_size = np.array(im.shape)

    # First, we resize the image
    im = _resive_img(im=im, output_size=scale_to, keep_aspect=keep_aspect)

    im = _convert_image_2_matrix_ranging_from_0_to_1(im=im)

    # add extra dimension if it doesn't have any
    im = add_third_dimension_if_needed(im=im, along_axis=2)

    # removing the alpha channel:
    if im.shape[2] == 4:
        im = im[:, :, :3]

    # fix colorspace:
    if im.shape[2] == 1 and colorspace == Colorspace.rgb:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 3 and colorspace == Colorspace.gray:
        im = 0.2989 * im[:, :, 0] + 0.5870 * im[:, :, 1] + 0.1140 * im[:, :, 2]
        im = im.reshape([*im.shape, 1])

    # Move color space to the first position
    im = im.transpose(2, 0, 1)

    return im, original_im_size


def load_image(im_path):
    success = False

    if backend.lower() == 'cv2':
        im = cv2.imread(im_path, flags=cv2.IMREAD_UNCHANGED)
        if im is None:
            im = np.zeros((10, 10, 3), dtype='uint8')
        else:
            success = True

    elif backend.lower() == 'skimage':
        try:
            im = io.imread(im_path)
            success = True
        except FileNotFoundError:
            im = np.zeros((10, 10, 3), dtype='uint8')

    else:
        raise ValueError(f'No image backend matching {backend}')

    return im, success


def _resive_img(im, output_size: Union[int, Union[Tuple[int, int], np.array]], keep_aspect: bool):
    """Resize image to proper size"""
    if type(output_size) == int:
        scale_to = output_size
        # First, we resize the image
        if keep_aspect:
            height = im.shape[0]
            width = im.shape[1]
            max_shape = max(width, height)
            if backend == 'cv2':
                output_size = (int(scale_to * width / max_shape), int(scale_to * height / max_shape))
            else:
                output_size = (int(scale_to * height / max_shape), int(scale_to * width / max_shape))
        else:
            output_size = (scale_to, scale_to)

    if backend == 'cv2':
        return cv2.resize(src=im, dsize=output_size, interpolation=cv2.INTER_LINEAR)
    
    if backend == 'skimage':
        return resize(image=im, output_shape=output_size, order=1, mode='constant', anti_aliasing=False)
    
    raise ValueError(f'Invalid image backend: {backend}')


def _convert_image_2_matrix_ranging_from_0_to_1(
    im: np.ndarray,
    target_dtype: Literal['float32', 'float64', 'float16'] = 'float32',
) -> np.ndarray:
    """
    Then we convert the image so that the values be in the range of [0-1]

    We use the dtype max/min for uints while for floats that should have been loaded
    using scikit-image we check that the range is between 0 and 1
    """
    if im.dtype in ['uint16', 'uint8', 'int16', 'int8']:
        max_val = np.iinfo(im.dtype).max
        min_val = np.iinfo(im.dtype).min
        
        im = im.astype(target_dtype)
        im -= min_val
        return im / (max_val - min_val)

    if im.dtype in ['float32', 'float64']:
        assert im.max() <= 1. and im.min() >= .0, \
            f'after resizing the image with skimage, the max and min are {im.max()}, {im.min()}'
        im = im.astype(target_dtype)
        return im

    raise ValueError(f'Unknown image format type: "{im.dtype}"')

