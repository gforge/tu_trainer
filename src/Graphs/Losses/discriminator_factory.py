from collections import OrderedDict
import math

import torch.nn as nn

from Graphs.Factories.Networks.neural_net import Neural_Net
from Graphs.Factories.Blocks.basic_conv_block import Basic_Conv_Block
from Graphs.Factories.Blocks.fully_connected import Fully_Connected
from GeneralHelpers import Singleton
from DataTypes import Consistency


class Discriminator_Factory(metaclass=Singleton):

    def __init__(self):
        # Private network storage - similar to the functionality in base_network_factory
        self.__neural_nets = {}

    def get_neural_net(self, head, optimizer_type: str):
        consistency = head.get_consistency()

        if consistency == Consistency.number:
            return self.__get_number_discriminator(head, optimizer_type)
        elif consistency == Consistency.d1:
            raise KeyError('1D disctiminator is not yet implemented')
        elif consistency == Consistency.d2:
            return self.__get_2D_discriminator(head, optimizer_type)
        elif consistency == Consistency.d3:
            raise KeyError('3D disctiminator is not yet implemented')
        else:
            raise KeyError(f'Unknown consistency type {consistency}')

    def __get_number_discriminator(self, head, optimizer_type):

        head_name = head.get_name()
        head_shape = head.get_tensor_shape()
        neural_net_name = self.__get_neural_net_name(tensor_shape=head_shape, head_name=head_name)
        if neural_net_name not in self.__neural_nets:
            layers = Fully_Connected(input_shape=head_shape, output_shape=[1], num_hidden_layers=1, dropout=False)

            self.__neural_nets[neural_net_name] = \
                Neural_Net(neural_net_name=neural_net_name,
                           layers=layers,
                           optimizer_type=optimizer_type,
                           load_from_batch=False)
        return self.__neural_nets[neural_net_name]

    def __get_2D_discriminator(self, head, optimizer_type):

        head_name = head.get_name()
        head_shape = head.get_tensor_shape()
        neural_net_name = self.__get_neural_net_name(tensor_shape=head_shape, head_name=head_name)
        if neural_net_name not in self.__neural_nets:
            input_w = head.get_width()
            input_h = head.get_height()
            input_c = head.get_channels()
            output_c = 32

            layer_index = 0

            min_wh = min(input_h, input_w)

            layers = OrderedDict([])

            while min_wh > 1:
                layer_index += 1
                # add one max pooling layer
                layers['conv_%d' % layer_index] = Basic_Conv_Block(input_c, output_c, 5)
                layers['max_pool_%d' % layer_index] = nn.MaxPool2d(2, 2)
                input_c = output_c
                output_c *= 2
                min_wh = math.floor((min_wh + 1) / 2)
                input_w = math.floor((input_w + 1) / 2)
                input_h = math.floor((input_h + 1) / 2)

            layers['fc_%d_%d_%d' % (input_w, input_h, input_c)] = Fully_Connected(
                input_shape=[input_w, input_h, input_c], output_shape=[1], num_hidden_layers=1, dropout=False)
            layers = nn.Sequential(layers)
            self.__neural_nets[neural_net_name] = \
                Neural_Net(neural_net_name=neural_net_name,
                           layers=layers,
                           optimizer_type=optimizer_type,
                           load_from_batch=False)
        return self.__neural_nets[neural_net_name]

    def __get_neural_net_name(self, tensor_shape, head_name):
        return 'discriminator_%s_%s' % (head_name, 'x'.join(str(d) for d in tensor_shape))
