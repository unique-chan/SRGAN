from torch import nn
from ResidualBlock import ResidualBlock
from UpsamlingBlock import UpsamlingBlock
import numpy as np


def list_to_sequential(layer_list):
    return nn.Sequential(*layer_list)


class Generator(nn.Module):
    def __init__(self, in_channels=3, scaling_factor=4, num_residual_blocks=16):
        super(Generator, self).__init__()                                                           # (in, out)
        num_upsampling_blocks = int(np.log2(scaling_factor))
        out_channels = 64
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(9, 9),       # ( 3, 64)
                      stride=1, padding=4, bias=False),  # why padding=4??
            nn.PReLU()
        )
        self.residual_blocks = list_to_sequential(
            [ResidualBlock(in_channels=out_channels, out_channels=out_channels)                     # (64, 64)
             for _ in range(num_residual_blocks)]
        )
        self.intermediate_block = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3),      # (64, 64)
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )  # for connecting residual blocks with upsampling blocks!
        self.upsampling_blocks = list_to_sequential(
            [UpsamlingBlock(in_channels=out_channels,                   # (64, 64 * num_upsampling_blocks ^ 2)
                            scaling_factor=num_upsampling_blocks)
             for _ in range(num_upsampling_blocks)]
        )
        self.output_block = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=(9, 9),
                      stride=1, padding=4, bias=False),
            # we might add extra non-linear activation here!
        )

    def forward(self, x):
        _out = self.input_block(x)
        out = self.residual_blocks(_out)
        out = _out + self.intermediate_block(out)
        out = self.upsampling_blocks(out)
        out = self.output_block(out)
        return out
