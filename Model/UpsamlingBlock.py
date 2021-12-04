from torch import nn


class UpsamlingBlock(nn.Module):
    def __init__(self, in_channels, scaling_factor, kernel_size=(3, 3), stride=1, padding=1):
        super(UpsamlingBlock, self).__init__()
        out_channels = in_channels * (scaling_factor ** 2)  # -> Why?
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(scaling_factor)
        self.p_relu = nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.p_relu(out)
        return out
