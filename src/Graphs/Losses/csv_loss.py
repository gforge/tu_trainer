from typing import List, Generic
from torch import Tensor

from DataTypes import CSV_Loss_Type

from .base_convergent_loss import Base_Convergent_Loss


class CSV_Loss(Generic[CSV_Loss_Type], Base_Convergent_Loss[CSV_Loss_Type]):
    """This is an abstract class for losses that are related to information
    withihn the CSV-file config.
    """

    @property
    def ignore_index(self) -> int:
        return self._cfgs.ignore_index

    @property
    def output_shape(self) -> List[int]:
        return self._cfgs.output_shape

    @property
    def to_each_view_its_own_label(self) -> bool:
        return self._cfgs.to_each_view_its_own_label

    @property
    def signal_to_noise_ratio(self) -> float:
        return self._cfgs.signal_to_noise_ratio

    @property
    def loss_weight(self) -> Tensor:
        return self._cfgs.loss_weight
