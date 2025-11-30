from typing import List, Optional

from DataTypes.Network.base import Network_Cfg_Base
from .raw import Network_Cfg_Cascade_Raw


class Network_Cfg_Cascade(Network_Cfg_Base, Network_Cfg_Cascade_Raw):
    output_shapes: Optional[List[int]] = None

    def __str__(self) -> str:
        return self.get_id()

    def get_id(self) -> str:
        """Combines the structure into a name

        Merges the components:
        - block
        - consistency
        - input_channels
        - blocks:
          - count
          - output_channels
          - kernel_size
        - head_name
        - purpose (encoder/decoder)
        - tail_name

        Returns:
            str: A dense id string with all components
        """
        base_def = f'{self.block_type}{self.consistency}_{self.input_channels}'
        block_body = '_'.join(f'{b.no_blocks}-{b.output_channels}-{b.kernel_size}' for b in self.blocks)
        post_def = f'{self.head_name}_{self.purpose}_{self.tail_name}'
        return f'{base_def}_{block_body}_{post_def}'

    def get_clean_name(self) -> str:
        """Retrieves a name safe for using in files

        Returns:
            str: A string without unexpected characters
        """
        safe_name = "".join([c for c in self.get_id() if c.isalnum() or c in ('_', '-')]).rstrip()
        if len(safe_name) == 0:
            raise ValueError(f'The clean safe name becomes empty with {str(self)}')
        return safe_name
