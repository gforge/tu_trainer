import torch
from torch import Tensor
from typing import Tuple, Union, Dict, Any
from operator import itemgetter
import numpy as np
import torchvision.utils as vutils

from GeneralHelpers.pytorch_wrapper import unwrap
from Utils.ImageTools import add_points_2_image
from UIs.scene_UI_manager import ResultId, SceneUIManager
from .writer_library import WriterLibrary
from .base_outcome_writer import BaseOutcomeWriter


def _add_points_2_image(images, points, radius, color: Tuple[int, int, int]):
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            images[i, j] = add_points_2_image(images[i, j], points=points[i, j], radius=radius, color=color)
    return images


class _DrawObject():
    """
    As we want to only do the unwrapping and converting when necessary we save the
    tensors to the tensorboard we have this object for saving references.
    """

    def __init__(self, imgs: Tensor, predicted: np.ndarray, target: np.ndarray) -> None:
        super().__init__()
        if isinstance(imgs, Tensor):
            imgs = imgs.detach().cpu()
        if isinstance(predicted, Tensor):
            predicted = predicted.detach().cpu()
        if isinstance(target, Tensor):
            target = target.detach().cpu()

        # Save detached versions without the backprop graph
        self.imgs = imgs
        self.predicted = predicted
        self.target = target

    def get_numpy_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        imgs_numpy = (unwrap(self.imgs, clone=True) * 255).astype(np.uint8)
        predicted_numpy = unwrap(self.predicted, clone=True)
        target_numpy = unwrap(self.target, clone=True)
        return imgs_numpy, predicted_numpy, target_numpy


Data2Store = Dict[str, _DrawObject]


class CoordinateWriter(BaseOutcomeWriter[Data2Store]):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scene_manager = SceneUIManager()
        self._target_color = (0, 0, 255)
        self._prediction_color = (255, 128, 0)

        # Elements specific to coordinate/line/independent_line/...
        self._visual_id = 'Visualization/Coordinates'
        self.results_output_name = 'coordinates'

    @property
    def __use_only_images_with_predictions4tensorboard(self):
        from global_cfgs import Global_Cfgs
        return Global_Cfgs().get('use_only_images_with_predictions4tensorboard', default=True)

    def _decorate_image(self, images: np.ndarray, predicted, target):
        images = _add_points_2_image(images=images, radius=2, points=target, color=self._target_color)
        images = _add_points_2_image(images=images, radius=1, points=predicted, color=self._prediction_color)
        return images

    def add_last_data(self, batch: Dict[str, Any]):
        result_id = self.scene_manager.get_result_id(ds=batch['dataset_name'],
                                                     exp=batch['experiment_name'],
                                                     task=batch['task_name'],
                                                     graph=batch['graph_name'])

        data_2_store: Data2Store = {}
        for (mod_id, data) in batch['results'].items():
            if self.results_output_name not in data:
                continue

            # Fetch and convert images only once
            imgs = batch['encoder_image']
            (predicted, target) = itemgetter('predicted', 'target')(data[self.results_output_name])
            data_2_store[mod_id] = _DrawObject(imgs=imgs, predicted=predicted, target=target)

        super().add_last_data(result_id=result_id, data=data_2_store, iteration=batch['iteration_counter'])

    def __get_grid(self, imgs: Union[Tensor, np.ndarray]) -> Tensor:
        if not isinstance(imgs, Tensor):
            imgs = torch.Tensor(imgs)

        h, w, colors = imgs.shape[::-1][0:3]
        imgs = imgs.reshape(-1, colors, w, h)

        return vutils.make_grid(imgs, normalize=True, scale_each=True)

    def __convert_data_2_decorated_grid(self, data_4_decoartion: _DrawObject):
        (imgs, predicted, target) = data_4_decoartion.get_numpy_data()

        # Convert to RGB, i.e. (batch, jitter, colors)
        if len(imgs.shape) == 5 and imgs.shape[2] == 1:
            if isinstance(imgs, np.ndarray):
                imgs = imgs.repeat(3, axis=2)
            else:
                imgs = imgs.repeat(1, 1, 3, 1, 1)
        imgs = self._decorate_image(images=imgs, predicted=predicted, target=target)
        if self.__use_only_images_with_predictions4tensorboard:
            has_target_4_image = (~np.isnan(target)).all(axis=1).all(axis=1).all(axis=1)
            imgs = imgs[has_target_4_image]

        if imgs.shape[0] == 0:
            return None

        return self.__get_grid(imgs=imgs)

    def _write_to_tensorboard(self, result_id: ResultId, iteration: int, data: Data2Store):
        writer = WriterLibrary().get(result_id=result_id)
        for mod_id, data_4_decoartion in data.items():
            grid = self.__convert_data_2_decorated_grid(data_4_decoartion=data_4_decoartion)
            if grid is None:
                return
            writer_id = f'{self._visual_id}/{mod_id}'
            writer.add_image(writer_id, grid, iteration)
