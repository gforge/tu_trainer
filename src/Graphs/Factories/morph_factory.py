import numpy as np
import torch.nn as nn
from typing import Any, Dict, List
from networkx import Graph
from DataTypes import Purpose, Network_Cfg_Morph_Single_Element

from .Networks.neural_net_set import Neural_Net_Set
from .Networks.neural_net import Neural_Net

from .Blocks.basic_conv_block import Basic_Conv_Block
from .Blocks.fully_connected import Fully_Connected
from .base_network_factory import Base_Network_Factory


class Morph_Factory(Base_Network_Factory):
    """ Morph factory merges two inputs into one, e.g. image and text report interpretations.
    """

    def get_neural_net(
        self,
        heads: Dict[str, Dict[str, Any]],
        tails: Dict[str, Dict[str, Any]],
        model_cfgs: Dict[str, Dict[str, Any]],
        optimizer_type: str,
        purpose: Purpose,
    ):
        assert len(tails) == 1
        tail_name = list(tails.keys())[0]
        tail = list(tails.values())[0]

        output_shape = tail['tensor_shape']
        output_consistency = tail['consistency']
        are_heads_input_modality = [
            'modality' in head and head['modality'].lower() == 'input'.lower() for head in heads
        ]

        neural_net_names = self.__get_neural_net_names(heads=heads, tail_name=tail_name, tail=tail, purpose=purpose)

        for i, head in enumerate(heads.values()):
            neural_net_name = neural_net_names[i]

            if neural_net_name not in self._neural_nets:
                init_net_args = {
                    'purpose': Purpose,
                    'neural_net_name': neural_net_name,
                    'optimizer_type': optimizer_type,
                }
                if purpose == Purpose.encoder:
                    init_net_args.update(input_name=head['encoder_name'],
                                         input_shape=head['tensor_shape'],
                                         input_consistency=head['consistency'],
                                         output_name=tail['encoder_name'],
                                         output_shape=output_shape,
                                         output_consistency=output_consistency,
                                         is_head_input_modality=are_heads_input_modality[i])

                if purpose == Purpose.decoder:
                    init_net_args.update(input_name=tail['decoder_name'],
                                         input_shape=output_shape,
                                         input_consistency=output_consistency,
                                         output_name=head['decoder_name'],
                                         output_shape=head['tensor_shape'],
                                         output_consistency=head['consistency'],
                                         is_head_input_modality=False)
                self._neural_nets[neural_net_name] = self.__init_neural_net(**init_net_args)

        return Neural_Net_Set([self._neural_nets[neural_net_name] for neural_net_name in neural_net_names])

    def __init_neural_net(
        self,
        input_shape,
        output_shape,
        input_consistency,
        output_consistency,
        input_name,
        output_name,
        optimizer_type,
        neural_net_name,
        is_head_input_modality,
    ):
        cfgs = Network_Cfg_Morph_Single_Element(name=neural_net_name,
                                                head_name=input_name,
                                                tail_name=output_name,
                                                input_shape=input_shape,
                                                output_shape=output_shape,
                                                consistency=input_consistency)

        if (input_shape == output_shape):
            layers = nn.Dropout(p=0, inplace=True)  # Just identity mapping
        elif (input_consistency == output_consistency):
            if (input_shape[0] == output_shape[0]):
                layers = nn.Dropout(p=0, inplace=True)  # Just identity mapping
            else:
                layers = Basic_Conv_Block(input_c=input_shape[0],
                                          output_c=output_shape[0],
                                          kernel_size=1,
                                          consistency=input_consistency,
                                          add_relu=not is_head_input_modality)

            # fix pooling size
        else:
            layers = Fully_Connected(input_shape=input_shape, output_shape=output_shape, num_hidden_layers=0)

        return Neural_Net(neural_net_cfgs=cfgs,
                          input_name=input_name,
                          output_name=output_name,
                          input_shape=input_shape,
                          output_shape=output_shape,
                          layers=layers,
                          optimizer_type=optimizer_type)

    def __get_neural_net_names(
        self,
        heads: Dict[str, Dict[str, Any]],
        tail: Dict,
        tail_name: str,
        purpose: Purpose,
    ) -> List[str]:
        output_shape_txt = 'x'.join(str(x) for x in tail['tensor_shape'])
        output_txt = f'{tail_name}_{output_shape_txt}'

        names: List[str] = []
        for head_name, head in heads.items():
            head_shape_txt = 'x'.join(str(x) for x in head['tensor_shape'])
            input_txt = f'{head_name}_{head_shape_txt}'

            names.append(f'morph_{input_txt}_{output_txt}_{purpose}'.lower())
        return names

    def update_modality_dims(
        self,
        neural_net_cfgs: Dict[str, Any],
        heads: List[str],
        tails: List[str],
        graph: Graph,
    ):
        tail = tails[0]
        tail_consistency = graph.nodes[tail]['consistency']
        tail_shape = None
        for head in heads:
            if graph.nodes[head]['consistency'] == tail_consistency:
                if tail_shape is None:
                    tail_shape = np.array(graph.nodes[head]['tensor_shape'].copy())
                else:
                    tail_shape = np.max([tail_shape, np.array(graph.nodes[head]['tensor_shape'].copy())], axis=0)

        graph.nodes[tail]['tensor_shape'] = list(tail_shape)
