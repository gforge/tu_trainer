from typing import Tuple
from .coordinate_writer import CoordinateWriter
from Utils.ImageTools import add_lines_2_image


def _add_lines_2_image(images, points, thickness, color: Tuple[int, int, int]):
    assert points.shape[2] % 2 == 0, 'Unexpected line shape'
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            images[i, j] = add_lines_2_image(images[i, j], points=points[i, j], thickness=thickness, color=color)
    return images


class LineWriter(CoordinateWriter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._visual_id = 'Visualization/Lines'
        self.results_output_name = 'lines'

    def _decorate_image(self, images, predicted, target):
        images = _add_lines_2_image(images=images, thickness=2, points=target, color=self._target_color)
        images = _add_lines_2_image(images=images, thickness=1, points=predicted, color=self._prediction_color)

        # Add points
        images = super()._decorate_image(images=images, predicted=predicted, target=target)

        return images
