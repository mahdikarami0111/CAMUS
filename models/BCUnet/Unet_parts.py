import copy
import math
from torchvision.transforms import ToPILImage, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.utils import _pair
import numpy as np


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.maxpool_conv(x)
        return self.relu(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1),
        nn.BatchNorm2d(out_channels),
    )


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=in_channels//32, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = self.up(x)
        outlayer = nn.Sigmoid()
        return outlayer(x)


class RecallCrossEntropy(nn.Module):
    def __init__(self, n_classes=2, ignore_index=255):
        super(RecallCrossEntropy, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # input (batch,n_classes,H,W)
        # target (batch,H,W)
        target = target.squeeze(1)
        pred = input.argmax(dim=1)
        idex = (pred != target).view(-1)

        # calculate ground truth counts
        gt_counter = torch.ones((self.n_classes,)).cuda()
        gt_idx, gt_count = torch.unique(target, return_counts=True)
        gt_idx, gt_count = gt_idx.int(), gt_count.int()

        # map ignored label to an existing one
        gt_count[gt_idx == self.ignore_index] = gt_count[1].clone()
        gt_idx[gt_idx == self.ignore_index] = 1
        gt_counter[gt_idx] = gt_count.float()

        # calculate false negative counts
        fn_counter = torch.ones((self.n_classes)).cuda()
        fn = target.view(-1)[idex]
        fn_idx, fn_count = torch.unique(fn, return_counts=True)
        fn_idx, fn_count = fn_idx.int(), fn_count.int()

        # map ignored label to an existing one
        fn_count[fn_idx == self.ignore_index] = fn_count[0].clone()

        fn_idx[fn_idx == self.ignore_index] = 1
        fn_counter[fn_idx] = fn_count.float()

        weight = fn_counter / gt_counter
        target = target.long()
        CE = F.cross_entropy(input, target, reduction='none', ignore_index=self.ignore_index)
        loss = weight[target] * CE
        return loss.mean()
