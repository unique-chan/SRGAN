from torch import nn


class ResidualBlock(nn.Module):  # for Generator (1)
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.p_relu = nn.PReLU()
        self.short_cut = nn.Sequential()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn(out)
        out = self.p_relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        out = self.short_cut(identity) + out
        return out


class UpsamlingBlock(nn.Module):  # for Generator (2)
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


class ConvBlock(nn.Module):  # for Discriminator
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)
