from collections import OrderedDict
from torch import nn

from DataTypes.enums import Consistency


class Basic_Conv_Block(nn.Module):
    """
    Basic conv block is just a convolution, followed by BatchNorm and ReLU.
    This is the basic block of networks like VGG net.
    """

    def __init__(
        self,
        input_c,
        output_c,
        kernel_size,
        consistency: Consistency,
        add_relu=True,
        groups=1,
    ):
        super().__init__()
        self.input_c = input_c
        self.output_c = output_c
        self.kernel_size = kernel_size
        self.consistency = consistency
        self.add_relu = add_relu

        if self.consistency == Consistency.d1:
            self.Conv = nn.Conv1d
            self.Norm = nn.BatchNorm1d
        elif self.consistency == Consistency.d2:
            self.Conv = nn.Conv2d
            self.Norm = nn.BatchNorm2d
        elif self.consistency == Consistency.d3:
            self.Conv = nn.Conv3d
            self.Norm = nn.BatchNorm3d
        else:
            raise KeyError(f'Unknown consistency : {self.consistency.value}')
        self.layers = self.get_layer()

    def get_layer(self):
        layers = OrderedDict({})
        if self.add_relu:
            layers['relu'] = nn.ReLU()
        num_groups = int(self.input_c)
        if num_groups % 32 == 0:
            num_groups = 32
        # layers['batch_norm'] = nn.GroupNorm(num_groups,self.input_c)
        layers['batch_norm'] = self.Norm(self.input_c)
        name = f'conv_{self.input_c}x{self.output_c}x{self.consistency.value}'.replace(".", "_")
        layers[name] =\
            self.Conv(self.input_c,
                      self.output_c,
                      kernel_size=self.kernel_size,
                      stride=1,
                      padding=self.kernel_size // 2,
                      bias=True)

        return nn.Sequential(layers)

    def forward(self, x):
        return self.layers(x)
