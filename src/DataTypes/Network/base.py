import abc

from DataTypes.general import BaseModelWithGet
from DataTypes.enums import Consistency, Purpose


class Network_Cfg_Base(BaseModelWithGet, abc.ABC):
    neural_net_type: str = ...
    "The type of network this belongs to"

    purpose: Purpose = ...
    "The core purpose of this network, often used in the name"

    head_name: str = ...
    "The name of the input data to the network"

    tail_name: str = ...
    "The name of the output data to the network"

    input_channels: int = ...
    "The number of input channels from the head input source"

    consistency: Consistency = ...
    "The dimensionality of the input data, 1D, 2D or 3D"

    is_head_input_modality: bool = ...
    "If this network is receiving direct input from raw data (i.e. in the bottom of the network)"

    @abc.abstractmethod
    def get_id(self) -> str:
        pass

    def get_clean_name(self) -> str:
        """Retrieves a name safe for using in files

        Returns:
            str: A string without unexpected characters
        """
        safe_name = "".join([c for c in self.get_id() if c.isalnum() or c == '_']).rstrip()
        if len(safe_name) == 0:
            raise ValueError(f'The clean safe name becomes empty with {str(self)}')
        return safe_name
