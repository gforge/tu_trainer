from torch import Tensor
import torchvision.utils as vutils
from typing import Dict, Any
from .writer_library import WriterLibrary
from operator import itemgetter
from .base_outcome_writer import BaseOutcomeWriter
from UIs.scene_UI_manager import ResultId


class _ReconstructionData():

    def __init__(self, original: Tensor, reconstructed: Tensor):
        self.original = original.detach().clone()
        self.reconstructed = reconstructed.detach().clone()


class ReconstructionWriter(BaseOutcomeWriter[_ReconstructionData]):

    def __get_vgrid(self, images: Tensor) -> Tensor:
        grid = vutils.make_grid(images, normalize=True, scale_each=True)
        return grid.detach().cpu().clone()

    def __convert_data_2_decorated_grids(self, data_4_grid: _ReconstructionData):
        original_grid = self.__get_vgrid(data_4_grid.original)
        reconstructed_grid = self.__get_vgrid(data_4_grid.reconstructed)
        return original_grid, reconstructed_grid

    def add_last_data(self, batch: Dict[str, Any]):
        if 'decoder_image' not in batch:
            return

        (original, decoded) = [v.detach().clone() for v in itemgetter('encoder_image', 'decoder_image')(batch)]
        result_id = self.__scene_manager.get_result_id(ds=batch['dataset_name'],
                                                       exp=batch['experiment_name'],
                                                       task=batch['task_name'],
                                                       graph=batch['graph_name'])
        data_2_store = _ReconstructionData(original=batch['encoder_image'], reconstructed=batch['decoder_image'])

        super().add_last_data(result_id=result_id, data=data_2_store, iteration=batch['iteration_counter'])

    def _write_to_tensorboard(self, result_id: ResultId, iteration: int, data: _ReconstructionData):
        writer = WriterLibrary().get(result_id=result_id)
        (original, reconstructed) = self.__convert_data_2_decorated_grids(data_4_grid=data)
        writer.add_image('Visualization/Original', original, iteration)
        writer.add_image('Visualization/Reconstruction', reconstructed, iteration)
