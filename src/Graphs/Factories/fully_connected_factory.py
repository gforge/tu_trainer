from typing import Any, Dict
from DataTypes import Purpose, Network_Cfg_Fully_Connected, Model_Cfg, OptimizerType
from .Networks.neural_net import Neural_Net
from .Blocks.fully_connected import Fully_Connected
from .base_network_factory import Base_Network_Factory


class Fully_Connected_Factory(Base_Network_Factory):
    """ Instantiates a fully connected (FC) network.
    """
    def get_neural_net(
        self,
        heads: Dict[str, Dict[str, Any]],
        tails: Dict[str, Dict[str, Any]],
        model_cfgs: Model_Cfg,
        optimizer_type: OptimizerType,
        purpose: Purpose,
    ):
        assert (len(heads) == 1)
        assert (len(tails) == 1)
        head_name = list(heads.keys())[0]
        tail_name = list(tails.keys())[0]

        head = list(heads.values())[0]
        tail = list(tails.values())[0]
        input_shape = head['tensor_shape']
        output_shape = tail['tensor_shape']

        is_head_input_modality = 'modality' in head and \
            head['modality'].lower() == 'input'.lower() and \
            not purpose == 'decoder'

        cfg_dict = model_cfgs.neural_net_cfgs.dict()
        cfg_dict.update({
            'purpose': purpose,
            'head_name': head_name,
            'head_shape': input_shape,
            'tail_name': tail_name,
            'tail_shape': output_shape,
            'consistency': head.get('consistency'),
            'input_channels': head.get('num_channels'),
            'is_head_input_modality': is_head_input_modality
        })

        cfgs = Network_Cfg_Fully_Connected(**cfg_dict)

        neural_net_name = str(cfgs)

        if neural_net_name not in self._neural_nets:
            init_net_args = {'cfgs': cfgs, 'optimizer_type': optimizer_type}
            if purpose == Purpose.encoder:
                init_net_args.update(input_name=head['encoder_name'],
                                     input_shape=input_shape,
                                     output_shape=output_shape,
                                     output_name=tail['encoder_name'])

            if purpose == Purpose.decoder:
                init_net_args.update(input_name=tail['decoder_name'],
                                     input_shape=output_shape,
                                     output_name=head['decoder_name'],
                                     output_shape=input_shape)
            self._neural_nets[neural_net_name] = self.__init_neural_net(**init_net_args)

        return self._neural_nets[neural_net_name]

    def __init_neural_net(
        self,
        cfgs: Network_Cfg_Fully_Connected,
        input_name,
        output_name,
        input_shape,
        output_shape,
        optimizer_type,
    ):
        layers = Fully_Connected(input_shape=input_shape,
                                 output_shape=output_shape,
                                 num_hidden_layers=cfgs.num_hidden,
                                 add_relu=cfgs.add_relu)

        return Neural_Net(neural_net_cfgs=cfgs,
                          input_name=input_name,
                          output_name=output_name,
                          input_shape=input_shape,
                          output_shape=output_shape,
                          layers=layers,
                          optimizer_type=optimizer_type)

    def update_modality_dims(
        self,
        neural_net_cfgs: dict,
        heads: list,
        tails: list,
        graph,
    ):
        return
