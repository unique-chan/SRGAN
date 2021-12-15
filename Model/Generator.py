from torch import nn
from Model.Block import ResidualBlock, list_to_sequential
from Model.Block import UpsamlingBlock
import numpy as np


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_residual_blocks=16):
        super(Generator, self).__init__()                                                           # (in, out)
        num_upsampling_blocks = 2
        out_channels = 64
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(9, 9),       # ( 3, 64)
                      stride=1, padding=4, bias=False),
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
        self.short_cut = nn.Sequential()

    def forward(self, x):
        identity = self.input_block(x)
        out = self.residual_blocks(identity)
        out = self.short_cut(identity) + self.intermediate_block(out)
        out = self.upsampling_blocks(out)
        out = self.output_block(out)
        return out
