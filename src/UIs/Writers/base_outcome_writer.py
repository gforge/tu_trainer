from collections import defaultdict
from collections.abc import Iterable
from UIs.scene_UI_manager import ResultId
from typing import Dict, TypeVar, Generic

CoreData = TypeVar('CoreData')


class _LastData(Generic[CoreData]):

    def __init__(self, data: CoreData, result_id: ResultId, iteration: int):
        self.data: CoreData = data
        self.iteration = iteration
        self.result_id = result_id


class BaseOutcomeWriter(Generic[CoreData]):

    def __init__(self):
        self._last_data: Dict[str, _LastData]
        self._reset_data()

    def _reset_data(self):
        self._last_data = defaultdict(dict)

    def _is_data_empty(self, data: CoreData):
        """
        We only want to work with data that is non-empty
        and therefore we check
        """
        if isinstance(data, dict) and len(data.keys()) == 0:
            return True

        if isinstance(data, Iterable) and len(data) == 0:
            return True

        return False

    def add_last_data(self, result_id: ResultId, data: CoreData, iteration: int):
        if self._is_data_empty(data=data):
            return self

        self._last_data[result_id.id] = _LastData(data=data, result_id=result_id, iteration=iteration)
        return self

    def add_to_tensorboard(self, result_id: ResultId):
        if result_id.id not in self._last_data:
            return

        data = self._last_data[result_id.id]
        self._write_to_tensorboard(result_id=data.result_id, iteration=data.iteration, data=data.data)

        del self._last_data[result_id.id]

    def flush_all_results_2_tensorboard(self):
        """
        Write and clear all results stored
        """
        for data in self._last_data.values():
            self._write_to_tensorboard(result_id=data.result_id, iteration=data.iteration, data=data.data)

        # Clear all and release saved data
        self._reset_data()

    def _write_to_tensorboard(self, result_id: ResultId, iteration: int, data: CoreData):
        raise NotImplementedError('The tensorboard writer has not been impemented')
