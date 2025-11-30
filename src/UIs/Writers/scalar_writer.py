import numpy as np
from pandas import DataFrame
from UIs.scene_UI_manager import ResultId
from .writer_library import WriterLibrary
from .base_outcome_writer import BaseOutcomeWriter


class ScalarWriter(BaseOutcomeWriter[DataFrame]):

    def _write_to_tensorboard(self, result_id: ResultId, iteration: int, data: DataFrame):
        writer = WriterLibrary().get(result_id=result_id)

        for measurement, data in data.items():
            for name in data.index:
                scalar = data[name]
                if not isinstance(scalar, (list, np.ndarray)) and not np.isnan(scalar):
                    if measurement[-4:] == 'loss':
                        # cap the loss as it is not interesting to view if it explodes
                        scalar = min(scalar, 20)
                    elif measurement[-3:] == 'auc':
                        # Values below 0.5 are meaningless
                        scalar = max(scalar, 0.5)

                    writer.add_scalar(f'{measurement}/{name}', scalar, iteration)
