from typing import Any, Dict, List, Tuple

from DataTypes import Network_Cfg_Pre_Defined
from DataTypes.Model.general import Model_Cfg
from DataTypes.enums import Purpose
from Graphs.Factories.Networks.neural_net import Neural_Net
from networkx import DiGraph
from torch.nn import Conv2d, Sequential, Module, SiLU, BatchNorm2d

from file_manager import File_Manager
from .base_network_factory import Base_Network_Factory


class Pre_Defined_Factory(Base_Network_Factory):
    """Class for retrieving pre-defined models

    When a model has a standard definition we may want to directly use that
    model where we only modify the bottom block to match image input together
    with dropping the final classifier so that we get the desired features.

    Args:
        Base_Network_Factory ([type]): This class has a memory of previously
                                       loaded networks so that we can share
                                       networks between network superstructures
    """

    def get_neural_net(
        self,
        heads: Dict,
        tails: Dict,
        model_cfgs: Model_Cfg,
        optimizer_type: str,
        purpose: Purpose,
    ) -> Neural_Net:
        """Retrieve a forward() network

        Args:
            heads (dict): The network after this (must be 1)
            tails (dict): The input data (must be 1)
            model_cfgs (dict): The configs for the network
            optimizer_type (str): Optimizer
            neural_net_type (str, optional): Must be 'encoder'.

        Returns:
            Neural_network: Returns a Neural_network class
        """
        assert len(heads) == 1, 'Pre-defined does not merge data'
        assert len(tails) == 1, 'Pre-defined does not fork data'
        assert purpose == Purpose.encoder, 'Decoder is not implemented for pre-defined models'
        head_name, head = list(heads.items())[0]
        tail_name, tail = list(tails.items())[0]
        input_shape = head['tensor_shape']

        cfg_dict = model_cfgs.neural_net_cfgs.dict()
        cfg_dict.update({
            'purpose': purpose,
            'head_name': head_name,
            'tail_name': tail_name,
            'input_channels': head['tensor_shape'][0],
            'consistency': head['consistency'],
            'is_head_input_modality': 'modality' in head and head['modality'].lower() == 'input'.lower()
        })
        cfgs = Network_Cfg_Pre_Defined(**cfg_dict)
        neural_net_name = str(cfgs)

        if str(cfgs) not in self._neural_nets:
            self._neural_nets[neural_net_name] = self.__get_encoder_model(cfgs=cfgs,
                                                                          input_shape=input_shape,
                                                                          optimizer_type=optimizer_type,
                                                                          head=head,
                                                                          tail=tail)

        return self._neural_nets[neural_net_name]

    def __get_encoder_model(self, cfgs: Network_Cfg_Pre_Defined, input_shape: List[int], optimizer_type: str,
                            head: Dict[str, Any], tail: Dict[str, Any]) -> Neural_Net:
        """Retrieves a model matching the config

        Args:
            cfgs (Pre_Defined_Network_Cfg): The config describing the network that we want to retrieve
            input_shape ([type]): The shape object of the input

        Raises:
            KeyError: KeyError

        Returns:
            [Neural_Net]: An extended torch.nn.Module object
        """
        layers: Module = None
        if cfgs.repo_or_dir.startswith('NVIDIA'):
            if cfgs.model.startswith('nvidia_resnet'):
                layers = self.__nvidia_resnet(cfgs=cfgs, input_shape=input_shape)
            if cfgs.model.startswith('nvidia_efficientnet'):
                layers = self.__nvidia_efficientnet(cfgs=cfgs, input_shape=input_shape)

        if layers is None:
            raise KeyError(f'Model {cfgs} not implemented')

        return Neural_Net(neural_net_cfgs=cfgs,
                          layers=layers,
                          optimizer_type=optimizer_type,
                          input_name=head['encoder_name'],
                          output_name=tail['encoder_name'],
                          input_shape=head["tensor_shape"],
                          output_shape=tail["tensor_shape"])

    def __nvidia_resnet(self, cfgs: Network_Cfg_Pre_Defined, input_shape) -> Neural_Net:
        model: Module = File_Manager().load_and_save_hub_network(repo_or_dir=cfgs.repo_or_dir,
                                                                 model=cfgs.model,
                                                                 pretrained=cfgs.pretrained)
        # Drop top classification layer and switch input layer
        first_layer: Conv2d = list(model.children())[0]
        if first_layer.in_channels != input_shape[0]:
            first_layer = Conv2d(in_channels=input_shape[0],
                                 out_channels=first_layer.out_channels,
                                 kernel_size=first_layer.kernel_size,
                                 stride=first_layer.stride,
                                 padding=first_layer.padding,
                                 dilation=first_layer.dilation,
                                 groups=first_layer.groups,
                                 padding_mode=first_layer.padding_mode,
                                 bias=first_layer.bias is not None)

        new_model = Sequential(first_layer, *(list(model.children())[1:-1]))

        return new_model

    def __nvidia_efficientnet(self, cfgs: Network_Cfg_Pre_Defined, input_shape) -> Neural_Net:
        model: Module = File_Manager().load_and_save_hub_network(repo_or_dir=cfgs.repo_or_dir,
                                                                 model=cfgs.model,
                                                                 pretrained=cfgs.pretrained)
        # Drop top classification layer and switch input layer
        first_layers: Tuple[Conv2d, BatchNorm2d, SiLU] = list(list(model.children())[0].children())
        first_layer = first_layers[0]
        if first_layer.in_channels != input_shape[0]:
            first_layer = Conv2d(in_channels=input_shape[0],
                                 out_channels=first_layer.out_channels,
                                 kernel_size=first_layer.kernel_size,
                                 stride=first_layer.stride,
                                 padding=first_layer.padding,
                                 dilation=first_layer.dilation,
                                 groups=first_layer.groups,
                                 padding_mode=first_layer.padding_mode,
                                 bias=first_layer.bias is not None)

        top_block = Sequential(first_layer, *first_layers[1:])
        new_model = Sequential(top_block, *(list(model.children())[1:-1]))

        return new_model

    def update_modality_dims(
        self,
        neural_net_cfgs: Network_Cfg_Pre_Defined,
        heads: List[str],
        tails: List[str],
        graph: DiGraph,
    ):
        """Update dimensions of expected inputs and output

        Automatically calculate the dimensions of the modalities if they are not specified in other config files.
        The reason that we calculate the modality dimension on the fly is that this way, we can dynamically make
        the graph and don't be bothered with the input, output sizes of tensors.

        Args:
            neural_net_cfgs (Network_Cfg_Pre_Defined): The network config for cascade network
            heads (List[str]): The names of the heads
            tails (List[str]): The names of the tails
            graph ([type]): The graph that we want to update
        """

        if neural_net_cfgs.repo_or_dir.startswith('NVIDIA'):
            if neural_net_cfgs.model.startswith('nvidia_resnet'):
                return self.__update_dims_4_adaptive_pool(graph=graph, heads=heads, tails=tails, channels=2048)

            if neural_net_cfgs.model.startswith('nvidia_efficientnet'):
                core_args = {'graph': graph, 'heads': heads, 'tails': tails, 'int_divisor': 30}
                if neural_net_cfgs.model.endswith('b0'):
                    core_args['channels'] = 1280
                elif neural_net_cfgs.model.endswith('b4'):
                    core_args['channels'] = 1792
                else:
                    raise KeyError(f'Unknown efficient net definition: {neural_net_cfgs.model}')

                return self.__update_dims_4_no_halvings(**core_args)

        raise KeyError(f'Model {neural_net_cfgs} not implemented')

    def __update_dims_4_adaptive_pool(self, graph: DiGraph, heads: List[str], tails: List[str], channels: int):
        graph.nodes[tails[0]]['tensor_shape'] = [channels, 1, 1]
        graph.nodes[tails[0]]['consistency'] = graph.nodes[heads[0]]['consistency']

    def __update_dims_4_no_halvings(
        self,
        graph: DiGraph,
        heads: List[str],
        tails: List[str],
        channels: int,
        int_divisor: int,
    ):
        "Reduction in size may be due to max_pool or stride 2 -> reduces the size by integer division"
        input_shape = graph.nodes[heads[0]]['tensor_shape']
        output_shape = [shape // int_divisor for shape in input_shape[-2:]]
        graph.nodes[tails[0]]['tensor_shape'] = [channels, *output_shape]
        graph.nodes[tails[0]]['consistency'] = graph.nodes[heads[0]]['consistency']
