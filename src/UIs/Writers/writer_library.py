from os import path
from torch.utils.tensorboard import SummaryWriter
from GeneralHelpers import Singleton
from UIs.scene_UI_manager import ResultId
from typing import Optional


class WriterLibrary(metaclass=Singleton):

    def __init__(self):
        self.writers = {}
        self.log_folder: Optional[str] = None

    def set_log_folder(self, log_folder: str):
        self.log_folder = log_folder

    def init_writer(self, result_id: ResultId, log_folder: str):
        sw_path = path.join(log_folder, 'tensorboard', result_id.path, result_id.name)
        self.writers[result_id.id] = SummaryWriter(sw_path)
        return self.writers[result_id.id]

    def get(self, result_id: ResultId, log_folder: str = None):
        if result_id.id in self.writers:
            return self.writers[result_id.id]

        if self.log_folder is None:
            raise KeyError(f'Requested {result_id.id} but log folder has not been set')

        return self.init_writer(result_id=result_id, log_folder=self.log_folder)
