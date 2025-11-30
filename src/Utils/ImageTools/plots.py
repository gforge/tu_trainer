import numpy as np
from typing import Tuple, Union, Iterable, Dict, Any
import cv2
import torch
from torch.functional import Tensor

from .pad_and_center_image import pad_and_center_image
from .color_helpers import convert_image_to_cv2_and_back_to_original_format, convert_image_into_cv2_format

def _shrink_to_range(value: float, min: int, max: int):
    if value < min:
        return min
    if value > max:
        return max
    return value
    
def _get_point_4_cv2(point, image_shape):
    w, h = image_shape[0:2]

    if torch.is_tensor(point):
        point = point.detach().cpu().numpy()

    x, y = np.multiply(point, (h, w)).astype(int)
    x = _shrink_to_range(x, min = 0, max = w)
    y = _shrink_to_range(y, min = 0, max = h)

    return x, y


@convert_image_to_cv2_and_back_to_original_format
def add_points_2_image(
        im,
        points,
        radius: int,
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = -1,
):
    for idx in range(int(points.shape[0])):
        point = points[idx, :]
        if np.isnan(point).any():
            continue

        x, y = _get_point_4_cv2(point=point, image_shape=im.shape)
        im = cv2.circle(im, (x, y), radius=radius, color=color, thickness=thickness)

    return im


@convert_image_to_cv2_and_back_to_original_format
def add_lines_2_image(
        im,
        points,
        thickness: int = 1,
        color: Tuple[int, int, int] = (0, 0, 255),
):
    assert points.shape[0] % 2 == 0, 'Invalid line matrix'

    cv2_img = im.copy()
    for no in range(int(points.shape[0] / 2)):
        idx = no * 2
        line_points = points[idx:(idx + 2), :]
        if np.isnan(line_points).any():
            continue

        x1, y1 = _get_point_4_cv2(point=line_points[0, :], image_shape=cv2_img.shape)
        x2, y2 = _get_point_4_cv2(point=line_points[1, :], image_shape=cv2_img.shape)

        cv2_img = cv2.line(cv2_img, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=thickness)

    return cv2_img


def _add_points_and_lines(im, points):
    im = add_lines_2_image(im=im, points=points, thickness=1)
    im = add_points_2_image(im=im, points=points, radius=2, color=(125, 0, 255))
    return im


def debug_plot_single_image(im, title: str = None) -> None:
    (im, _) = convert_image_into_cv2_format(im=im)
    import matplotlib.pyplot as plt
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.show()


def plot_points_on_image(
    im,
    points,
    title: str = None,
    debug: bool = True,
):
    im = _add_points_and_lines(im=im, points=points)

    cv2_img, _ = convert_image_into_cv2_format(im)
    (transformed_img, _) = pad_and_center_image(cv2_img)
    debug_plot_single_image(im=transformed_img, title=title)


def debug_plot_org_and_transformed_points(
    org_image: np.array,
    images_with_meta: list,
    main_title: str = None,
):
    """
    A helper for checking that the output matches the expected format
    """
    images = [{
        'name': 'original',
        'im': org_image
    }, *[{
        **data, 'im': _add_points_and_lines(im=data['im'], points=data['points'])
    } for data in images_with_meta]]

    import matplotlib.pyplot as plt
    columns = len(images)
    fig = plt.figure(figsize=(4 * columns, 4))
    rows = 1
    ax = []
    for i in range(columns * rows):
        ax.append(fig.add_subplot(rows, columns, i + 1))
        title = f'Name: {images[i]["name"]}'
        ax[i].set_title(title)
        cv2_img, _ = convert_image_into_cv2_format(images[i]['im'].copy())
        if cv2_img.dtype == np.float32:
            cv2_img *= 2**8
            cv2_img = cv2_img.astype(np.uint8)
        plt.imshow(cv2_img)

    if main_title is not None:
        fig.suptitle(main_title)
    plt.show()


def debug_plot_images(
    org_image,
    images: Union[Iterable[Tensor], Tensor],
    main_title: str = None,
) -> None:
    if isinstance(images, Tensor):
        images = [images[i] for i in range(images.shape[0])]

    all_images = (org_image, *images)
    columns = len(all_images)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4 * columns, 4))

    rows = 1
    ax = []
    for i in range(columns * rows):
        ax.append(fig.add_subplot(rows, columns, i + 1))
        if i == 0:
            ax[i].set_title('Original')
        else:
            ax[i].set_title(f'Version {i}')

        cv2_img, _ = convert_image_into_cv2_format(all_images[i].copy())
        if cv2_img.dtype == np.float32:
            cv2_img *= 2**8
            cv2_img = cv2_img.astype(np.uint8)
        plt.imshow(cv2_img)

    if main_title is not None:
        fig.suptitle(main_title)
    plt.show()


def debug_write_batch_encoded_images_and_quit(batch: Dict[str, Any], out_folder: str = None):
    # Check images - seem ok
    import os
    import sys
    from global_cfgs import Global_Cfgs
    if out_folder is None:
        out_folder = os.path.join(Global_Cfgs().log_folder, 'img_debug')
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)

    for i in range(batch['encoder_image'].shape[0]):
        for ii in range(batch['encoder_image'].shape[1]):
            img = batch['encoder_image'][i, ii, 0] * 255
            cv2.imwrite(
                os.path.join(out_folder, f'test{i}_{ii}.png'),
                img.reshape(*img.shape, 1).cpu().numpy(),
            )

    print(f'Wrote images to {out_folder}')
    sys.exit()
