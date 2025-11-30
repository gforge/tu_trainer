from collections import OrderedDict
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from DataTypes import Purpose, Network_Cfg_Cascade_Raw, Network_Cfg_Cascade
from DataTypes.Model.general import Model_Cfg
from DataTypes.enums import Consistency

from .Networks.neural_net import Neural_Net
from .Blocks.basic_conv_block import Basic_Conv_Block
from .Blocks.resnet_basic_block import ResNet_Basic_Block
from .Blocks.custom_dropout import Custom_Dropout
from .base_network_factory import Base_Network_Factory


class Cascade_Factory(Base_Network_Factory):
    """ Responsible for generating a nn.Sequential (cascade of neural networks)
    """
    def get_neural_net(
        self,
        heads: dict,
        tails: dict,
        model_cfgs: Model_Cfg,
        optimizer_type: str,
        purpose: Purpose,
    ):
        assert len(heads) == 1, 'Cascade does not merge data'
        assert len(tails) == 1, 'Cascade does not fork data'
        head_name, head = list(heads.items())[0]
        tail_name, tail = list(tails.items())[0]
        head_shape = head['tensor_shape']
        tail_shape = tail['tensor_shape']

        cfg_dict = model_cfgs.neural_net_cfgs.dict()
        cfg_dict.update({
            'purpose': purpose,
            'head_name': head_name,
            'tail_name': tail_name,
            'input_channels': head_shape[0],
            'consistency': head['consistency'],
            'is_head_input_modality': 'modality' in head and head['modality'].lower() == 'input'.lower()
        })
        cfgs = Network_Cfg_Cascade(**cfg_dict)

        neural_net_name = str(cfgs)

        if neural_net_name not in self._neural_nets:
            if purpose == Purpose.encoder:
                self._neural_nets[neural_net_name] = \
                    self.__get_encoder(input_c=cfgs.input_channels,
                                       input_name=head['encoder_name'],
                                       output_name=tail['encoder_name'],
                                       input_shape=head_shape,
                                       output_shape=tail_shape,
                                       cfgs=cfgs,
                                       optimizer_type=optimizer_type)

            if purpose == Purpose.decoder:
                self._neural_nets[neural_net_name] = \
                    self.__get_decoder(output_c=cfgs.input_channels,
                                       input_name=tail['decoder_name'],
                                       output_name=head['decoder_name'],
                                       input_shape=tail_shape,
                                       output_shape=head_shape,
                                       cfgs=cfgs,
                                       optimizer_type=optimizer_type)

        return self._neural_nets[neural_net_name]

    def __get_encoder(
        self,
        input_c: int,
        input_name: str,
        output_name: str,
        input_shape: np.ndarray,
        output_shape: np.ndarray,
        cfgs: Network_Cfg_Cascade,
        optimizer_type: str,
    ):
        output_shapes = self.__get_output_shapes(shape=input_shape,
                                                 num_blocks=len(cfgs),
                                                 add_max_pool_after_each_block=cfgs.add_max_pool_after_each_block)[1:]
        cfgs.output_shapes = output_shapes

        self.MaxPool = _get_max_pool(consistency=cfgs.consistency)
        Block = _get_block(block_type=cfgs.block_type)

        layers = OrderedDict({})
        if cfgs.custom_dropout is not None:
            layers['custom_dropout'] = Custom_Dropout(cfgs.custom_dropout)

        # If the input is raw data then we don't want to do a ReLu on that data
        add_relu = not cfgs.is_head_input_modality
        for i, block in enumerate(cfgs.blocks):
            output_sh = output_shapes[i]
            for j in range(block.no_blocks):
                # Retain the original size until the last block where we adapt to the
                # next output shape
                block_output_c = input_c
                if j == block.no_blocks - 1:
                    block_output_c = block.output_channels

                # This should be f'{input_c}_o{block_output_c}_k{kernel_size}' but
                # since we want to load old trained networks we keep this description
                block_desc = f'{input_c}_o{block_output_c}_k{block.kernel_size}'
                block_id = f'l{i}_i{j}_j{block_desc}'
                layers[block_id] = Block(input_c=input_c,
                                         output_c=block_output_c,
                                         kernel_size=block.kernel_size,
                                         consistency=cfgs.consistency,
                                         add_relu=add_relu)
                add_relu = True

            input_c = block.output_channels
            layers[f'dn_{i}'] = self.MaxPool(output_sh)

        return Neural_Net(neural_net_cfgs=cfgs,
                          input_name=input_name,
                          output_name=output_name,
                          input_shape=input_shape,
                          output_shape=output_shape,
                          layers=nn.Sequential(layers),
                          optimizer_type=optimizer_type)

    def __get_decoder(
        self,
        output_c,
        input_name: str,
        output_name: str,
        input_shape: np.ndarray,
        output_shape: np.ndarray,
        cfgs: Network_Cfg_Cascade,
        optimizer_type: str,
    ):
        input_c = cfgs.blocks[-1].output_channels
        kernel_sizes = [b.kernel_size for b in reversed(cfgs.blocks)]
        block_counts = [b.no_blocks for b in reversed(cfgs.blocks)]
        block_output_cs = [b.output_channels for b in list(reversed(cfgs.blocks))[:-1]]
        block_output_cs.append(output_c)

        output_shapes = list(
            reversed(
                self.__get_output_shapes(shape=output_shape,
                                         num_blocks=len(cfgs),
                                         add_max_pool_after_each_block=cfgs.add_max_pool_after_each_block)))[1:]

        block_infos = zip(block_counts, block_output_cs, kernel_sizes, output_shapes)

        Block = _get_block(block_type=cfgs.block_type)

        layers = OrderedDict({})
        for i, (block_count, output_c, kernel_size, output_sh) in enumerate(block_infos):
            layers[f'dn_{i}'] = torch.nn.Upsample(output_sh)

            for j in range(block_count):
                block_input_c = output_c
                if j == 0:
                    block_input_c = input_c

                block_desc = f'{input_c}_o{output_c}_k{kernel_size}'
                block_id = f'l{i}_i{j}_j{block_desc}'
                layers[block_id] = Block(input_c=block_input_c,
                                         output_c=output_c,
                                         kernel_size=kernel_size,
                                         consistency=cfgs.consistency,
                                         add_relu=True)
            input_c = output_c

        return Neural_Net(neural_net_cfgs=cfgs,
                          input_name=input_name,
                          output_name=output_name,
                          input_shape=input_shape,
                          output_shape=output_shape,
                          layers=nn.Sequential(layers),
                          optimizer_type=optimizer_type)

    def update_modality_dims(
        self,
        neural_net_cfgs: Network_Cfg_Cascade_Raw,
        heads: List[str],
        tails: List[str],
        graph,
    ):
        """Update dimensions of expected inputs and output

        Automatically calculate the dimensions of the modalities if they are not specified in other config files.
        The reason that we calculate the modality dimension on the fly is that this way, we can dynamically make
        the graph and don't be bothered with the input, output sizes of tensors.

        Args:
            neural_net_cfgs (Network_Cfg_Cascade_Core): The network config for cascade network
            heads (List[str]): The names of the heads
            tails (List[str]): The names of the tails
            graph ([type]): The graph that we want to update
        """
        output_c = neural_net_cfgs.blocks[-1].output_channels
        output_shape = self.__get_output_shapes(
            shape=graph.nodes[heads[0]]['tensor_shape'],
            num_blocks=len(neural_net_cfgs),
            add_max_pool_after_each_block=neural_net_cfgs.add_max_pool_after_each_block,
        )
        graph.nodes[tails[0]]['tensor_shape'] = [output_c, *output_shape[-1]]
        graph.nodes[tails[0]]['consistency'] = graph.nodes[heads[0]]['consistency']

    def __get_output_shapes(
        self,
        shape: np.ndarray,
        num_blocks: int,
        add_max_pool_after_each_block: bool,
    ) -> List[np.ndarray]:
        # every block will be followed by a max pooling operation,
        shape = shape[1:]
        all_shapes = [shape]
        if isinstance(shape, (list, tuple)):
            shape = np.array(shape, dtype=int)
        for _ in range(num_blocks):
            if add_max_pool_after_each_block:
                shape = shape // 2
                shape[shape < 1] = 1
            all_shapes.append(list(shape))
        return all_shapes


def _get_max_pool(consistency: Consistency) -> Union[nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d]:
    if consistency == Consistency.d1:
        return nn.AdaptiveMaxPool1d

    if consistency == Consistency.d2:
        return nn.AdaptiveMaxPool2d

    if consistency == Consistency.d3:
        return nn.AdaptiveMaxPool3d

    raise KeyError(f'Unknown consistency: {consistency}')


def _get_block(block_type: str) -> Union[Basic_Conv_Block, ResNet_Basic_Block]:
    if block_type.lower() == 'Basic'.lower():
        return Basic_Conv_Block
    if block_type.lower() == 'ResNetBasic'.lower():
        return ResNet_Basic_Block

    raise KeyError(f'Unknown block type {block_type}')
