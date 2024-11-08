import torch.nn as nn
import torch
from utils.wavelet import DWT_2D, IDWT_2D
import copy


class DownSamplingDWT(nn.Module):
    def __init__(self, in_channels, wavename='haar'):
        super(DownSamplingDWT, self).__init__()
        self.dwt = DWT_2D(wavename=wavename, in_channels=in_channels)

    @staticmethod
    def bayes_thresh(lh, hl, hh):
        channels = lh.shape[1]
        batch_size = lh.shape[0]
        height = hh.shape[2]
        width = hh.shape[3]
        sigma = torch.median(torch.abs(hh.view(batch_size, channels, height * width)), dim=2)[0] / .6745

        sigma_y = torch.stack((lh, hl, hh), dim=1).view(batch_size, 3, channels, height * width)
        sigma_y = torch.mean(torch.pow(sigma_y, 2), dim=-1)

        sigma = sigma.unsqueeze(dim=1)
        sigma_x = torch.sqrt(torch.maximum(sigma_y - torch.pow(sigma, 2), torch.tensor(0.00001)))
        T = torch.pow(sigma, 2) / sigma_x
        T = torch.maximum(T, torch.tensor(0.0000000001).cuda())
        return T

    @staticmethod
    def denoise_wavelet(lh, hl, hh, T):
        channels = lh.shape[1]
        batch_size = lh.shape[0]
        height = hh.shape[2]
        width = hh.shape[3]
        coeffs = torch.stack((lh, hl, hh), dim=1).view(batch_size, 3, channels, height * width)
        m = torch.abs(coeffs)
        T = T.unsqueeze(-1)
        denom = m + (m < T).float()
        gain = torch.maximum(m-T, torch.tensor(0))/denom
        coeffs = (coeffs * gain).view(batch_size, 3, channels, height, width)
        lh = coeffs[:, 0, :, :, :].squeeze(1)
        hl = coeffs[:, 1, :, :, :].squeeze(1)
        hh = coeffs[:, 2, :, :, :].squeeze(1)

        return lh, hl, hh

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        T = self.bayes_thresh(LH, HL, HH)
        LH, HL, HH = self.denoise_wavelet(LH, HL, HH, T)
        return LL, LH, HL, HH, input


class UpSamplingIDWT(nn.Module):
    def __init__(self, in_channels, wavename='haar'):
        super(UpSamplingIDWT, self).__init__()
        self.in_channels = in_channels
        self.idwt = IDWT_2D(wavename=wavename, in_channels=in_channels)

    def forward(self, LL, LH, HL, HH, feature_map):
        return torch.cat((self.idwt(LL, LH, HL, HH), feature_map), dim=1)


class WUNet(nn.Module):
    def __init__(self, features, num_classes=1, in_channels=3, init_weights=True, mother_wavelet='haar'):
        super(WUNet, self).__init__()
        self.classifier_seg = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1, padding=0),
        )
        self.wavename = mother_wavelet
        self.features = features
        self.in_channels = in_channels
        self.encoder = nn.ModuleList(self.init_encoder())
        self.decoder = nn.ModuleList(self.init_decoder())
        self.dr = nn.Dropout2d(0.4)

    def init_encoder(self):
        blocks = []
        dropouts = [nn.Dropout2d(0.2), nn.Dropout2d(0.2), nn.Dropout2d(0.2), nn.Dropout2d(0.2)]
        i = 0
        in_channels = self.in_channels
        for index, l in enumerate(self.features):
            if l == 'M':
                blocks.append(DownSamplingDWT(wavename=self.wavename, in_channels=self.features[index-1]))
                blocks.append(dropouts[i])
                i += 1
            else:
                blocks.append(nn.Sequential(
                    nn.Conv2d(in_channels, l, kernel_size=3, padding=1),
                    nn.BatchNorm2d(l),
                    nn.ReLU(inplace=True)
                ))
                in_channels = l
        return blocks

    def init_decoder(self):
        blocks = []
        out_channel_final = 64
        features = copy.deepcopy(self.features)
        features.reverse()
        for index, l in enumerate(features):

            if index != len(features) - 1:
                out_channel = features[index+1]
            else:
                out_channel = out_channel_final

            if out_channel == 'M':
                out_channel = features[index+2]

            if l == 'M':
                blocks.append(UpSamplingIDWT(wavename=self.wavename, in_channels=features[index+1]))
            else:
                if features[index - 1] == 'M':
                    l = 2 * l
                blocks.append(nn.Sequential(
                    nn.Conv2d(l, out_channel, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(inplace=True)
                ))

        return blocks

    def forward(self, x):
        wavelet_features = []
        ll = x
        index = 0
        for module in self.encoder:
            if isinstance(module, DownSamplingDWT):
                ll, LH, HL, HH, residual = module(ll)
                wavelet_features.insert(0, (LH, HL, HH, residual))
            else:
                ll = module(ll)

        for module in self.decoder:
            if isinstance(module, UpSamplingIDWT):
                features = wavelet_features[index]
                ll = module(ll, features[0], features[1], features[2], features[3])

                index += 1
            else:
                ll = module(ll)

        ll = self.dr(ll)
        return self.classifier_seg(ll)
