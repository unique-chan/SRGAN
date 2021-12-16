from torch import nn
from Model.Block import ConvBlock, list_to_sequential


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, conv_in_out_chs_strides=None):
        super(Discriminator, self).__init__()
        self.linear_width = 1024
        if not conv_in_out_chs_strides:
            conv_in_out_chs_strides = [(64, 64, 2), (64, 128, 1), (128, 128, 2),
                                       (128, 256, 1), (256, 512, 1), (512, 512, 2)]
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3),
                      stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )
        self.conv_blocks = list_to_sequential(
            [ConvBlock(in_channels=in_ch, out_channels=out_ch, stride=stride)
             for in_ch, out_ch, stride in conv_in_out_chs_strides]
        )
        self.output_block = nn.Sequential(
            nn.LazyLinear(out_features=self.linear_width),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.input_block(x)
        out = self.conv_blocks(out)
        out = out.view(-1, self.linear_width)
        out = self.output_block(out)
        return out
