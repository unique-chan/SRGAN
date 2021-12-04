from torch import nn
from ConvBlock import ConvBlock
import numpy as np


def list_to_sequential(layer_list):
    return nn.Sequential(*layer_list)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, scaling_factor=4, num_residual_blocks=16):
        super(Discriminator, self).__init__()
        in_features = 10  # <- i have to implement !!!
        conv_in_out_chs_strides = [(in_channels, 64, 2), (64, 128, 1), (128, 128, 2),
                                   (128, 256, 1), (256, 512, 1), (512, 512, 2)]
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3),
                      stride=1, padding=1, bias=False),     # why padding = 1??
            nn.LeakyReLU(0.2)
        )
        self.conv_blocks = list_to_sequential(
            [ConvBlock(in_channels=in_ch, out_channels=out_ch, stride=stride)
             for in_ch, out_ch, stride in conv_in_out_chs_strides]
        )
        self.output_block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=1024),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.input_block(x)
        out = self.conv_blocks(out)
        out = self.output_block(out)
        return out
