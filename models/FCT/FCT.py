import torch.nn as nn
import numpy as np
import torch
from torchvision.ops.stochastic_depth import StochasticDepth


class ConvAttention(nn.Module):
    def __init__(self, channels, num_heads, img_size, proj_drop=0.0, kernel_size=3, stride_kv=1, stride_q=1,
                 padding_kv="same", padding_q="same", attention_bias=True):
        super(ConvAttention, self).__init__()

        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = num_heads
        self.proj_drop = proj_drop

        self.layer_q = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride_q, padding_q, bias=attention_bias, groups=channels),
            nn.ReLU(),
        )
        self.layernorm_q = nn.LayerNorm(channels, eps=1e-5)

        self.layer_k = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride_kv, padding_kv, bias=attention_bias, groups=channels),
            nn.ReLU(),
        )
        self.layernorm_k = nn.LayerNorm(channels, eps=1e-5)

        self.layer_v = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride_kv, padding_kv, bias=attention_bias, groups=channels),
            nn.ReLU(),
        )
        self.layernorm_v = nn.LayerNorm(channels, eps=1e-5)

        self.attention = nn.MultiheadAttention(embed_dim=channels,
                                               bias=attention_bias,
                                               batch_first=True,
                                               dropout=self.proj_drop,
                                               num_heads=self.num_heads)

    def _build_projection(self, x, mode):
        if mode == 0:
            x1 = self.layer_q(x)
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_q(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif mode == 1:
            x1 = self.layer_k(x)
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_k(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif mode == 2:
            x1 = self.layer_v(x)
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_v(x1)
            proj = x1.permute(0, 3, 1, 2)
        else:
            raise ValueError("invalid value for projection build")

        return proj

    def get_qkv(self, x):
        q = self._build_projection(x, 0)
        k = self._build_projection(x, 1)
        v = self._build_projection(x, 2)

        return q, k, v

    def forward(self, x):
        q, k, v = self.get_qkv(x)
        q = q.view(q.shape[0], q.shape[1], q.shape[2] * q.shape[3])
        k = k.view(k.shape[0], k.shape[1], k.shape[2] * k.shape[3])
        v = v.view(v.shape[0], v.shape[1], v.shape[2] * v.shape[3])
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        x1 = self.attention(query=q, value=v, key=k, need_weights=False)
        x1 = x1[0].permute(0, 2, 1)
        x1 = x1.view(x1.shape[0], x1.shape[1], np.sqrt(
            x1.shape[2]).astype(int), np.sqrt(x1.shape[2]).astype(int))
        return x1


class Transformer(nn.Module):
    def __init__(self, out_channels, num_heads, dpr, img_size, proj_drop=0.0, attention_bias=True,
                 padding_q="same", padding_kv="same", stride_kv=1, stride_q=1):
        super(Transformer, self).__init__()
        self.attention_output = ConvAttention(channels=out_channels,
                                              num_heads=num_heads,
                                              img_size=img_size,
                                              proj_drop=proj_drop,
                                              padding_q=padding_q,
                                              padding_kv=padding_kv,
                                              stride_kv=stride_kv,
                                              stride_q=stride_q,
                                              attention_bias=attention_bias,
                                              )
        self.stochastic_depth = StochasticDepth(dpr, mode='batch')
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same")
        self.layernorm = nn.LayerNorm([out_channels, img_size, img_size])
        self.wide_focus = WideFocus(out_channels, out_channels)

    def forward(self, x):
        x1 = self.attention_output(x)
        x1 = self.stochastic_depth(x1)
        x2 = self.conv1(x1) + x

        x3 = self.layernorm(x2)
        x3 = self.wide_focus(x3)
        x3 = self.stochastic_depth(x3)

        out = x3 + x2
        return out


class WideFocus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WideFocus, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.layer_dilation2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same", dilation=2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.layer_dilation3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same", dilation=3),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer_dilation2(x)
        x3 = self.layer_dilation3(x)
        added = x1 + x2 + x3
        x_out = self.layer4(added)
        return x_out


class BlockEncoderSkip(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr, img_size):
        super(BlockEncoderSkip, self).__init__()
        self.layernorm = nn.LayerNorm([in_channels, img_size, img_size])
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )
        # image size /= 2
        self.layer3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d((2, 2))
        )
        self.trans = Transformer(out_channels, att_heads, dpr, img_size // 2)

    def forward(self, x, scale_img):
        x1 = self.layernorm(x)
        x1 = torch.cat((self.layer1(scale_img), x1), axis=1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.trans(x1)
        return x1


class BlockEncoderNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr, img_size):
        super(BlockEncoderNoSkip, self).__init__()
        self.layernorm = nn.LayerNorm([in_channels, img_size, img_size])
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )
        # img_size => img_size // 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d((2, 2))
        )
        self.trans = Transformer(out_channels, att_heads, dpr, img_size // 2)

    def forward(self, x):
        x = self.layernorm(x)
        x1 = self.layer1(x)
        x1 = self.layer2(x1)
        x1 = self.trans(x1)
        return x1


class BlockDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, att_heads, dpr, img_size):
        super(BlockDecoder, self).__init__()

        self.layernorm = nn.LayerNorm([in_channels, img_size, img_size])
        self.upsample = nn.Upsample(scale_factor=2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.trans = Transformer(out_channels, att_heads, dpr, img_size * 2)

    def forward(self, x, skip):
        x1 = self.layernorm(x)
        x1 = self.upsample(x1)
        x1 = self.layer1(x1)
        x1 = torch.cat((skip, x1), axis=1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        out = self.trans(x1)
        return out


class DsOut(nn.Module):
    def __init__(self, in_channels, out_channels, img_size):
        super(DsOut, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.layernorm = nn.LayerNorm([in_channels, img_size * 2, img_size * 2], eps=1e-5)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding="same"),
        )

    def forward(self, x):
        x1 = self.upsample(x)
        x1 = self.layernorm(x1)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        out = self.conv3(x1)

        return out


class FCT(nn.Module):
    def __init__(self, img_size):
        super(FCT, self).__init__()
        self.drp_out = 0.3
        self.img_size = img_size

        att_heads = [2, 4, 8, 16, 32, 16, 8, 4, 2]
        filters = [32, 64, 128, 256, 512, 256, 128, 64, 32]
        # att_heads = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        # filters = [8, 16, 32, 64, 128, 64, 32, 16, 8]
        blocks = len(filters)
        stochastic_depth_rate = 0.5
        dpr = [x for x in np.linspace(0, stochastic_depth_rate, blocks)]
        self.scale_img = nn.AvgPool2d(2, 2)
        self.block_1 = BlockEncoderNoSkip(1, filters[0], att_heads[0], dpr[0], self.img_size)
        self.block_2 = BlockEncoderSkip(filters[0], filters[1], att_heads[1], dpr[1], self.img_size // 2)
        self.block_3 = BlockEncoderSkip(filters[1], filters[2], att_heads[2], dpr[2], self.img_size // 4)
        self.block_4 = BlockEncoderSkip(filters[2], filters[3], att_heads[3], dpr[3], self.img_size // 8)
        self.block_5 = BlockEncoderNoSkip(filters[3], filters[4], att_heads[4], dpr[4], self.img_size // 16)
        self.block_6 = BlockDecoder(filters[4], filters[5], att_heads[5], dpr[5], self.img_size // 32)
        self.block_7 = BlockDecoder(filters[5], filters[6], att_heads[6], dpr[6], self.img_size // 16)
        self.block_8 = BlockDecoder(filters[6], filters[7], att_heads[7], dpr[7], self.img_size // 8)
        self.block_9 = BlockDecoder(filters[7], filters[8], att_heads[8], dpr[8], self.img_size // 4)

        self.ds7 = DsOut(filters[6], 1, self.img_size // 8)
        self.ds8 = DsOut(filters[7], 1, self.img_size // 4)
        self.ds9 = DsOut(filters[8], 1, self.img_size // 2)

    def forward(self, x):
        scale_img_2 = self.scale_img(x)
        scale_img_3 = self.scale_img(scale_img_2)
        scale_img_4 = self.scale_img(scale_img_3)

        x = self.block_1(x)
        skip1 = x

        x = self.block_2(x, scale_img_2)
        skip2 = x

        x = self.block_3(x, scale_img_3)
        skip3 = x

        x = self.block_4(x, scale_img_4)
        skip4 = x

        x = self.block_5(x)

        x = self.block_6(x, skip4)

        x = self.block_7(x, skip3)
        skip7 = x

        x = self.block_8(x, skip2)
        skip8 = x

        x = self.block_9(x, skip1)
        skip9 = x

        out7 = self.ds7(skip7)
        out8 = self.ds8(skip8)
        out9 = self.ds9(skip9)

        return out7, out8, out9


def init_weights(m):
    """
    Initialize the weights
    """
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)









