from pydantic import BaseModel, PositiveInt


class Network_Cfg_Cascade_Block(BaseModel):
    output_channels: PositiveInt
    "The block output channels"

    kernel_size: PositiveInt = 3
    "The size of the kernels"

    no_blocks: PositiveInt = 1
    "The number of blocks within each section"
