import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=(2, 2), stride=2, padding=0)
        self.conv = ConvBlock(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.e1 = EncoderBlock(input_channels, 64)
        self.e2 = EncoderBlock(64, 128)
        self.dr2 = nn.Dropout2d(0.2)
        self.e3 = EncoderBlock(128, 256)
        self.dr3 = nn.Dropout2d(0.2)
        self.e4 = EncoderBlock(256, 512)
        self.dr4 = nn.Dropout2d(0.3)

        self.b = ConvBlock(512, 1024)
        self.dr5 = nn.Dropout2d(0.2)
        self.dr6 = nn.Dropout2d(0.4)

        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)

        self.outputs = nn.Conv2d(64, output_channels, kernel_size=(1, 1), padding=0)

    def forward(self, X):
        s1, p1 = self.e1(X)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        p3 = self.dr3(p3)
        s4, p4 = self.e4(p3)
        p4 = self.dr4(p4)

        b = self.b(p4)
        b = self.dr5(b)

        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        d4 = self.dr6(d4)

        Y = self.outputs(d4)

        return Y


