import torch
from torch import nn
import torch.nn.functional as F


class SEBLock(nn.Module):
    def __init__(self, C, r=16):
        super().__init__()

        self.aap = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(C, C//r)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(C//r, C)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.aap(x)
        out = self.flatten(out)

        out = self.relu(self.linear1(out))
        out = self.sigmoid(self.linear2(out))

        out = out[:, :, None, None]

        res = x * out

        return res

# --------------------------------------------------------------------------------------------------------------------------------------------


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bnorm=False, upsample=True):
        super().__init__()

        if upsample == True:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.upsample = nn.Identity()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        if bnorm == False:
            self.bnorm = nn.Identity()
        elif bnorm == True:
            self.bnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.upsample(x)
        out = F.leaky_relu(self.bnorm(self.conv(x)), 0.2)
        return out

# --------------------------------------------------------------------------------------------------------------------------------------------

class DeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm=False, upsample=False):
        super().__init__()

        if upsample == True:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.upsample = nn.Identity()

        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        if norm == True:
            self.bnorm = nn.InstanceNorm2d(out_channels)
        else:
            self.bnorm = nn.Identity()

    def forward(self, x):
        x = self.upsample(x)
        out = F.leaky_relu(self.bnorm(self.conv(x)), 0.2)
        return out