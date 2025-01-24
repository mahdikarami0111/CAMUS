import torch.nn as nn
import torch
import torch.nn.functional as F
from models.UneXt.UneXt import Block as SeparableConv2d
from torchvision.ops.stochastic_depth import StochasticDepth


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dp_rate=0.):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.drop_path = StochasticDepth(dp_rate, mode='batch')

    def forward(self, x):
        return self.drop_path(self.double_conv(x))


class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim=3, dp_rt=0.):
        super(SeparableConvBlock, self).__init__()
        self.bottleneck = nn.Sequential(
            *[SeparableConv2d(in_channels, dp_rt) for i in range(dim)],
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.bottleneck(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dp_rate=0.):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dp_rate=dp_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class SeparableDown(nn.Module):
    def __init__(self, in_channels, out_channels, dp_rate=0.):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            SeparableConvBlock(in_channels, out_channels, dp_rt=dp_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class SeparableUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, dp_rate=0.):
        super(SeparableUp, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = SeparableConvBlock(in_channels * 2, out_channels, dp_rt=dp_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = SeparableConvBlock(in_channels, out_channels, dp_rt=dp_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        x = x1 if x2 is None else torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, dp_rate=0.):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels * 2, out_channels, in_channels, dp_rate=dp_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dp_rate=dp_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        x = x1 if x2 is None else torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, dims, dp_rate=0.):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, dp_rate, len(dims) - 1)]
        for i in range(len(dims) - 1):
            self.layers.append(Down(dims[i], dims[i+1], dp_rate=dp_rates[i]))

    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != (len(self.layers) - 1):
                skips.append(x)
        return x, skips


class SeparableEncoder(nn.Module):
    def __init__(self, dims, dp_rate=0.):
        super(SeparableEncoder, self).__init__()
        self.layers = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, dp_rate, len(dims) - 1)]
        for i in range(len(dims) - 1):
            self.layers.append(SeparableDown(dims[i], dims[i+1], dp_rate=dp_rates[i]))

    def forward(self, x):
        skips = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != (len(self.layers) - 1):
                skips.append(x)
        return x, skips


class Decoder(nn.Module):
    def __init__(self, dims, dp_rate=0.):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, dp_rate, len(dims) - 1)]
        dp_rates.reverse()
        for i in range(len(dims) - 1):
            self.layers.append(Up(dims[i], dims[i+1], dp_rate=dp_rates[i]))

    def forward(self, x, skips):
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, skips[i])
            outs.append(x)

        return x, outs


class SeparableDecoder(nn.Module):
    def __init__(self, dims, dp_rate=0.):
        super(SeparableDecoder, self).__init__()
        self.layers = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, dp_rate, len(dims) - 1)]
        dp_rates.reverse()
        for i in range(len(dims) - 1):
            self.layers.append(SeparableUp(dims[i], dims[i+1], dp_rate=dp_rates[i]))

    def forward(self, x, skips):
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, skips[i])
            outs.append(x)

        return x, outs

# enc1 = [64, 128, 256, 512, 1024]
# b1 = 512
# dec1 = [512, 256, 128]
# b2 = 128
# enc2 = [128, 256, 512, 1024]
# b3 = 512
# dec2 = [512, 256, 128, 64]


class WnetV3(nn.Module):
    def __init__(self, in_channels, num_classes, enc_dims, bottleneck_dims, dec_dims):
        super(WnetV3, self).__init__()
        self.stem = DoubleConv(in_channels=in_channels, out_channels=enc_dims[0][0])

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        enc1_dims = enc_dims[0]
        enc2_dims = enc_dims[1]
        dec1_dims = dec_dims[0]
        dec2_dims = dec_dims[1]

        self.enc1 = SeparableEncoder(enc1_dims)

        self.bn1 = SeparableConvBlock(enc1_dims[-1], bottleneck_dims[0], 5, 0.1)

        self.dec1 = SeparableDecoder(dec1_dims)

        self.bn2 = DoubleConv(dec1_dims[-1] * 2, bottleneck_dims[1], dec1_dims[-1] // 2)

        self.enc2 = Encoder(enc2_dims)

        self.bn3 = SeparableConvBlock(enc2_dims[-1], bottleneck_dims[2], 7, 0.1)

        self.dec2 = Decoder(dec2_dims)

        self.out = nn.Conv2d(in_channels=dec2_dims[-1], out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        stem = self.stem(x)
        x, skips_1 = self.enc1(stem)
        x = self.bn1(x)
        x, outs = self.dec1(x, [skips_1[i] for i in range(len(skips_1)-1, 0, -1)])

        bn = self.up(x)
        bn = torch.cat([bn, skips_1[0]], dim=1)
        bn2 = self.bn2(bn)

        x, skips_2 = self.enc2(bn2)
        skips_2.reverse()
        skips_2.append(bn2)
        skips_2.append(stem)
        x = self.bn3(x)
        x, outs = self.dec2(x, skips_2)
        x = self.out(x)

        return x, outs
