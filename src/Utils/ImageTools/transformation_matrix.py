import math
import numpy as np


def get_random_spatial_transform_matrix(
    input_size,
    output_size,
    augmentation_index=0,
    allow_random_stretch: bool = False,
):
    # With deep learning, data augmentation increases the accuracy of the model,
    # which is why we like to have as few copies of the same sample as possible
    # during the traning process. Hence we ignore the augmentation_index

    # If -in the future- it is discovered that we should augment the data
    # during the traning, we can use augmentation_index accordingly.
    rotation = [0, 90][np.random.randint(2)]
    if np.random.rand() > .5:
        rotation += np.random.normal(0, 15)

    c_x, c_y = np.random.normal(0, .2, 2)
    if allow_random_stretch:
        # Warning - this needs some fixing of spatial transform and thus never really activated
        stretch = np.random.normal(1, .1, 2),
    else:
        stretch = (1 - abs(c_x), 1 - abs(c_y))

    # Mirror horizontally slightly less than random as we want
    # the system to know left-right from the direction
    return _get_cropped_patch(input_size=input_size,
                              output_size=output_size,
                              center_ratio=(c_x, c_y),
                              rotation=rotation,
                              stretch=stretch,
                              horizontal_mirror=np.random.rand() > .8,
                              vertical_mirror=np.random.rand() > .8)


def get_fixed_spatial_transform_matrix(input_size, output_size, augmentation_index=0):
    fix_param = _get_fix_spatial_transform_params(augmentation_index)
    return _get_cropped_patch(input_size=input_size,
                              output_size=output_size,
                              center_ratio=fix_param['center_ratio'],
                              rotation=fix_param['rotation'],
                              stretch=fix_param['stretch'],
                              horizontal_mirror=fix_param['horizontal_mirror'],
                              vertical_mirror=fix_param['vertical_mirror'])


def _get_fix_spatial_transform_params(augmentation_index):  # NOSONAR
    # good number for augmentations are :
    # 1: the image itself
    # 4: images flipped horizontally and vertically
    # 12: the 4 images are rotated -10, 0 and 10 degrees
    # 108: corner cropped of those 12 images
    center_ratio = [0, -1 / 4, 1 / 4]
    rotation = [0, -10, 10]
    horizontal_mirror = [False, True]
    vertical_mirror = [False, True]
    cnt = 0
    while True:
        for c_x in center_ratio:
            for c_y in center_ratio:
                for r in rotation:
                    for h in horizontal_mirror:
                        for v in vertical_mirror:
                            cnt += 1
                            if augmentation_index < cnt:
                                return {
                                    'center_ratio': (c_x, c_y),
                                    'stretch': (1 - abs(c_x), 1 - abs(c_y)),
                                    'rotation': r,
                                    'horizontal_mirror': h,
                                    'vertical_mirror': v
                                }
        # if we reach to this stage, it means our augmentation size has been
        # bigger than 108, so we make the cropped images smaller and the angle of rotations more:
        rotation = [r * 1.5 for r in rotation]
        center_ratio = [c * .1 for c in center_ratio]
        # and we go back to the "while True" loop to sample another 108 images.
        # Honestly, I don't think we will ever need to augment and image more
        # than 12 times but I just add support for infinite data augmentation


def _get_cropped_patch(
        input_size,
        output_size,
        center_ratio=(0, 0),
        rotation=0,
        stretch=(1, 1),
        horizontal_mirror=False,
        vertical_mirror=False,
):

    def _get_corners(shape):
        return np.array([
            [0, 0, 1],
            [shape[1], 0, 1],
            [shape[1], shape[0], 1],
            [0, shape[0], 1],
        ], dtype='float32').T

    input_corners = _get_corners(shape=input_size)
    output_corners = _get_corners(shape=output_size)

    input_center = np.mean(input_corners, axis=1)
    output_center_in_original_image = (np.array([*center_ratio, 0]) + 1) * input_center
    output_center_in_cropped_image = np.mean(output_corners, axis=1)

    output_corners_in_original_image = output_corners.copy()
    output_corners_in_original_image -= output_center_in_cropped_image[:, np.newaxis]
    output_corners_in_original_image *= np.array([*stretch, 0])[:, np.newaxis]
    output_corners_in_original_image[2, :] = 1
    output_corners_in_original_image = np.dot(_rotate(rotation), output_corners_in_original_image)
    output_corners_in_original_image += output_center_in_original_image[:, np.newaxis]
    output_corners_in_original_image[2, :] = 1

    if (horizontal_mirror):
        output_corners = output_corners[:, [1, 0, 3, 2]]
    if (vertical_mirror):
        output_corners = output_corners[:, [3, 2, 1, 0]]

    # numpy can solve AX=B, here we want to solve M . original = cropped
    # To do so, we must reformulate the equation:
    # (M . original)' = cropped' # => original' . M' = cropped'

    M = np.linalg.lstsq(output_corners_in_original_image.T, output_corners.T, rcond=-1)[0].T
    M[np.abs(M) < 1e-4] = 0  # because it is visually more appealing

    return M


# I'm only using the rotate function at the moment, the rest are implemented inside
# get_cropped_patch. But I leave the functions here in case we later on decide to
# test something


def _rotate(
        theta,
        M=np.eye(3, dtype='float32'),
):
    theta = theta * math.pi / 180
    rt = np.array(
        [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]],
        dtype='float32',
    )
    rot_M = rt
    return np.dot(rot_M, M)
