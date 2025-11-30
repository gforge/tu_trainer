from typing import Any, Dict, List
import torch.nn as nn

from DataTypes import Network_Cfg_Fork_Single_Element, Purpose

from .Networks.neural_net_set import Neural_Net_Set
from .Networks.neural_net import Neural_Net

from .Blocks.basic_conv_block import Basic_Conv_Block
from .base_network_factory import Base_Network_Factory


class Fork_Factory(Base_Network_Factory):
    """ A fork splits a pathway int 1:n where n is the number of subpaths.

    This is usually used after interpreting an exam using a core network. E.g.
    an exam may have one sub-net for body-part, knee-fractures, hip-fracture, etc.
    """

    def get_neural_net(
        self,
        heads: dict,
        tails: dict,
        model_cfgs: dict,
        optimizer_type: str,
        purpose: Purpose,
    ):

        assert len(heads) == 1, 'A fork may only have a single input source'
        head_name, head = list(heads.items())[0]
        consistency = head['consistency']
        is_head_input_modality = 'modality' in head and \
            head['modality'].lower() == 'input'.lower()

        neural_net_names = self.__get_neural_net_names(head_name=head_name,
                                                       head=head,
                                                       tails=tails,
                                                       neural_net_type=purpose)

        for i, tail in enumerate(tails.values()):
            neural_net_name = neural_net_names[i]

            if neural_net_name not in self._neural_nets:
                init_net_args = {
                    'purpose': purpose,
                    'consistency': consistency,
                    'neural_net_name': neural_net_name,
                    'optimizer_type': optimizer_type,
                }
                if purpose == Purpose.encoder:
                    init_net_args.update(input_name=head['encoder_name'],
                                         input_c=head['num_channels'],
                                         input_shape=head['tensor_shape'],
                                         output_name=tail['encoder_name'],
                                         output_c=tail['num_channels'],
                                         output_shape=tail['tensor_shape'],
                                         is_head_input_modality=is_head_input_modality)

                if purpose == Purpose.decoder:
                    init_net_args.update(input_name=tail['decoder_name'],
                                         input_c=tail['num_channels'],
                                         input_shape=tail['tensor_shape'],
                                         output_name=head['decoder_name'],
                                         output_c=head['num_channels'],
                                         output_shape=head['tensor_shape'],
                                         is_head_input_modality=False)

                self._neural_nets[neural_net_name] = self.__init_neural_net(**init_net_args)

        return Neural_Net_Set([self._neural_nets[neural_net_name] for neural_net_name in neural_net_names])

    def __init_neural_net(
        self,
        input_c: int,
        output_c: int,
        consistency: bool,
        input_name: str,
        output_name: str,
        input_shape,
        output_shape,
        optimizer_type,
        neural_net_name,
        is_head_input_modality,
        purpose: Purpose,
    ):
        cfgs = Network_Cfg_Fork_Single_Element(input_channels=input_c,
                                               output_channels=output_c,
                                               head_name=input_name,
                                               tail_name=output_name,
                                               name=neural_net_name,
                                               consistency=consistency,
                                               purpose=purpose,
                                               is_head_input_modality=is_head_input_modality)

        if (input_c == output_c):
            layers = nn.Dropout(p=0, inplace=True)  # Just identity mapping
        else:
            layers = Basic_Conv_Block(input_c=input_c,
                                      output_c=output_c,
                                      kernel_size=1,
                                      consistency=consistency,
                                      add_relu=cfgs.add_relu)
        return Neural_Net(neural_net_cfgs=cfgs,
                          input_name=input_name,
                          output_name=output_name,
                          input_shape=input_shape,
                          output_shape=output_shape,
                          layers=layers,
                          optimizer_type=optimizer_type)

    def __get_neural_net_names(
        self,
        head_name,
        head,
        tails: Dict[str, Dict[str, Any]],
        neural_net_type,
    ) -> List[str]:
        names: List[str] = []

        input_txt = f'{head_name}_{head["num_channels"]}'
        for output_name, tail in tails.items():
            output_txt = f'{output_name}_{tail["num_channels"]}'
            names.append(f'fork_{input_txt}_{output_txt}_{neural_net_type}'.lower())
        return names

    def update_modality_dims(
        self,
        neural_net_cfgs: dict,
        heads: list,
        tails: list,
        graph,
    ):
        tensor_shape = graph.nodes[heads[0]]['tensor_shape'].copy()

        for i in range(len(tails)):
            if 'num_channels' in graph.nodes[tails[i]]:
                tensor_shape[0] = graph.nodes[tails[i]]['num_channels']
            graph.nodes[tails[i]]['tensor_shape'] = tensor_shape.copy()
            graph.nodes[tails[i]]['consistency'] =\
                graph.nodes[heads[0]]['consistency']
