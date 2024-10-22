import torch.nn as nn
import torch
from utils.wavelet import DWT_2D, IDWT_2D
from collections import OrderedDict
from itertools import islice
import operator


class DownSamplingDWT(nn.Module):
    def __init__(self, in_channels, wavename='haar'):
        super(DownSamplingDWT, self).__init__()
        self.dwt = DWT_2D(wavename=wavename, in_channels=in_channels)

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        return LL, LH, HL, HH, input


class UpSamplingIDWT(nn.Module):
    def __init__(self, in_channels, wavename='haar'):
        super(UpSamplingIDWT, self).__init__()
        self.in_channels = in_channels
        self.idwt = IDWT_2D(wavename=wavename, in_channels=in_channels)

    def forward(self, LL, LH, HL, HH, feature_map):
        return torch.cat((self.idwt(LL, LH, HL, HH), feature_map), dim=1)


class My_Sequential(nn.Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    若某个模块输出多个数据，只将第一个数据往下传
    """

    def __init__(self, *args):
        super(My_Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(My_Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input):
        self.output = []
        for module in self._modules.values():
            input = module(input)
            if isinstance(input, tuple):
                assert len(input) == 4 or len(input) == 2 or len(input) == 5
                self.output.append(input[1:])
                input = input[0]
        if self.output != []:
            return input, self.output
        else:
            return input


class My_Sequential_re(nn.Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    若某个模块输出多个数据，只将第一个数据往下传
    """

    def __init__(self, *args):
        super(My_Sequential_re, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        self.output = []

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(My_Sequential_re, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input):
        LL = input[0]
        index = 1
        for module in self._modules.values():
            if isinstance(module, UpSamplingIDWT):
                LH = input[index]
                HL = input[index + 1]
                HH = input[index + 2]
                feature_map = input[index + 3]
                LL = module(LL, LH, HL, HH, feature_map=feature_map)
                index += 4
            else:
                LL = module(LL)
        return LL


class WSegNetVGG(nn.Module):
    def __init__(self, features, num_classes=1, init_weights=True, wavename=None):
        super(WSegNetVGG, self).__init__()
        self.features = features[0]
        self.decoders = features[1]
        self.classifier_seg = nn.Sequential(
            # nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            # nn.ReLU(True),
            nn.Conv2d(64, num_classes, kernel_size=1, padding=0),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        xx = self.features(x)
        x, [(LH1, HL1, HH1, x1), (LH2, HL2, HH2, x2), (LH3, HL3, HH3, x3), (LH4, HL4, HH4, x4),
            (LH5, HL5, HH5, x5)] = xx
        x_ = [x, LH5, HL5, HH5, x5, LH4, HL4, HH4, x4, LH3, HL3, HH3, x3, LH2, HL2, HH2, x2, LH1, HL1, HH1, x1]
        x = self.decoders(x_)
        x = self.classifier_seg(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if (m.in_channels != m.out_channels or m.out_channels != m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    print('Not initializing')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def __str__(self):
        return 'U_Net_VGG'


def make_w_layers(cfg, batch_norm=False, wavename='haar'):
    encoder = []
    in_channels = 3
    temp = None
    for index, v in enumerate(cfg):
        if v != 'M':
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                encoder += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                encoder += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        elif v == 'M':
            encoder += [DownSamplingDWT(wavename=wavename, in_channels=cfg[index - 1])]
            temp = cfg[index - 1]
    encoder = My_Sequential(*encoder)

    decoder = []
    cfg.reverse()
    out_channels_final = 64
    flag = False
    for index, v in enumerate(cfg):
        if index != len(cfg) - 1:
            out_channels = cfg[index + 1]
        else:
            out_channels = out_channels_final
        if out_channels == 'M':
            out_channels = cfg[index + 2]
        if v == 'M':
            decoder += [UpSamplingIDWT(wavename=wavename, in_channels=cfg[index+1])]
        else:
            if cfg[index - 1] == 'M':
                v = 2 * v
            conv2d = nn.Conv2d(v, out_channels, kernel_size=3, padding=1)
            if batch_norm:
                decoder += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
            else:
                decoder += [conv2d, nn.ReLU(inplace=True)]
    decoder = My_Sequential_re(*decoder)
    return encoder, decoder
