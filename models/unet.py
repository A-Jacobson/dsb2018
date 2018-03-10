import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def pad_to_factor(image, factor=16):
    """Pads an image so that it is divisible by a given factor.
    Set factor to 2 ** number of pooling layers in architecture).
    Prevents size mismatches in Unet.
    """
    height, width = image.size()[-2:]
    h = math.ceil(height / factor) * factor
    w = math.ceil(width / factor) * factor
    pad_height = h - height
    pad_width = w - width
    top = math.ceil(pad_height / 2)
    bottom = math.floor(pad_height / 2)
    left = math.ceil(pad_width / 2)
    right = math.floor(pad_width / 2)
    if any((top, bottom, left, right)):
        return F.pad(image, mode='reflect', pad=(left, right, top, bottom))
    return image


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batch_norm(x)
        x = F.relu(self.conv2(x))
        x = self.batch_norm(x)
        return x


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1 = DoubleConvBlock(3, 32)
        self.conv2 = DoubleConvBlock(32, 64)
        self.conv3 = DoubleConvBlock(64, 128)
        self.conv4 = DoubleConvBlock(128, 256)

        self.conv5 = DoubleConvBlock(256, 512)

        self.conv6 = DoubleConvBlock(768, 256)
        self.conv7 = DoubleConvBlock(384, 128)
        self.conv8 = DoubleConvBlock(192, 64)
        self.conv9 = DoubleConvBlock(96, 32)
        self.conv10 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = pad_to_factor(x, factor=2 ** 4)
        c1 = self.conv1(x)
        x = self.pool(c1)
        c2 = self.conv2(x)
        x = self.pool(c2)
        c3 = self.conv3(x)
        x = self.pool(c3)
        c4 = self.conv4(x)
        x = self.pool(c4)

        bottle = self.conv5(x)

        x = self.up(bottle)
        x = self.conv6(torch.cat([x, c4], 1))
        x = self.up(x)
        x = self.conv7(torch.cat([x, c3], 1))
        x = self.up(x)
        x = self.conv8(torch.cat([x, c2], 1))
        x = self.up(x)
        x = self.conv9(torch.cat([x, c1], 1))
        x = self.conv10(x)
        return x
