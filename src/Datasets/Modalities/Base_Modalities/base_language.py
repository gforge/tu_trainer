from abc import ABCMeta

from DataTypes import Modality_Text_Cfg

from .base_sequence import Base_Sequence


class Base_Language(Base_Sequence[Modality_Text_Cfg], metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.char_to_ix = {}
        for i in range(len(self.dictionary)):
            self.char_to_ix[self.dictionary[i]] = i

        self.set_channels(len(self.dictionary))
        self.set_width(self.sequence_length)

    @property
    def case_sensitive(self) -> bool:
        return self._cfgs.case_sensitive

    @property
    def discard_numbers(self) -> bool:
        return self._cfgs.discard_numbers

    @property
    def sequence_length(self) -> int:
        return self._cfgs.sentence_length

    @property
    def dictionary(self) -> str:
        return self._cfgs.dictionary
