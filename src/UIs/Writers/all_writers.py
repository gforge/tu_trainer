from typing import List
from GeneralHelpers import Singleton
from UIs.scene_UI_manager import ResultId
from .base_outcome_writer import BaseOutcomeWriter
from .scalar_writer import ScalarWriter
from .coordinate_writer import CoordinateWriter
from .reconstruction_writer import ReconstructionWriter
from .line_writer import LineWriter
from .independent_line_writer import IndependentLineWriter


class AllWriters(metaclass=Singleton):

    def __init__(self):
        # Basic scalars, e.g. AUC, loss, sensitivity
        self.scalar_outcome = ScalarWriter()

        # Reconstructed image from decoder
        self.reconstruction = ReconstructionWriter()

        # Visual outputs with points, lines drawn onto the images
        self._visuals = [CoordinateWriter(), LineWriter(), IndependentLineWriter()]

    def __all_writers(self) -> List[BaseOutcomeWriter]:
        return [self.scalar_outcome, self.reconstruction, *self._visuals]

    def add_to_tensorboard(self, result_id: ResultId, iteration: int):
        args = {'result_id': result_id, 'iteration': iteration}
        [w.add_to_tensorboard(**args) for w in self.__all_writers()]

    def add_last_data_2_visuals(self, batch):
        [v.add_last_data(batch=batch) for v in self._visuals]

    @property
    def special_results_names(self):
        return [v.results_output_name for v in self._visuals]

    def flush_all_results_2_tensorboard(self):
        """
        This functions makes sure that everything that has been stored for
        later outputting is pushed into the tensorboard and that the storage
        memory is cleared.
        """
        [w.flush_all_results_2_tensorboard() for w in self.__all_writers()]
