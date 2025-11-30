from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel

from DataTypes.custom_dropout import Custom_Dropout_Cfg
from .block import Network_Cfg_Cascade_Block


class Network_Cfg_Cascade_Raw(BaseModel):
    neural_net_type: Literal['Cascade']

    blocks: List[Network_Cfg_Cascade_Block]

    add_max_pool_after_each_block: bool = True
    "Add max pool after each block"

    block_type: Literal['Basic', 'ResNetBasic']
    "The type of blocks used, Basic or ResNetBasic"

    custom_dropout: Optional[Custom_Dropout_Cfg] = None
    "If we want to block section of the input"

    def __init__(self, **data: Dict[str, Any]) -> None:
        data = self.__fix_old_format_compatibility(data)
        super().__init__(**data)

    def __fix_old_format_compatibility(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert old way of writing block in multiple lists of the same length to a single block list
        """
        old_config_names = ('block_output_cs', 'kernel_sizes', 'block_counts')
        if 'blocks' in data:
            assert not any([v in data for v in old_config_names]), \
                'You have both old and new cascade config format present'
            return data

        assert 'block_output_cs' in data, 'Old format should have both block_counts and block_output_cs defined'
        assert 'block_counts' in data, 'Old format should have both block_counts and block_output_cs defined'
        if 'kernel_sizes' not in data:
            data['kernel_sizes'] = [3 for _ in range(len(data['block_output_cs']))]

        data['blocks'] = [{
            'output_channels': oc,
            'kernel_size': ks,
            'no_blocks': no
        } for (oc, ks, no) in zip(data['block_output_cs'], data['kernel_sizes'], data['block_counts'])]

        del data['block_output_cs']
        del data['block_counts']
        del data['kernel_sizes']
        return data

    def __len__(self) -> int:
        return len(self.blocks)
