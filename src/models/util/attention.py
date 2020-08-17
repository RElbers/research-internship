import torch
import torch.nn as nn


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.pseudo_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1)
        )

        self.scale = None

    def forward(self, x):
        y_avg = self.pseudo_mlp(self.avg_pool(x))
        y_max = self.pseudo_mlp(self.max_pool(x))
        y = y_avg + y_max

        self.scale = torch.sigmoid(y)
        return x * self.scale


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.norm = nn.BatchNorm2d(2)

        self.scale = None

    def forward(self, x):
        y_avg = torch.mean(x, dim=1, keepdim=True)
        y_max, _ = torch.max(x, dim=1, keepdim=True)

        y = torch.cat([y_avg, y_max], dim=1)
        y = self.conv(y)

        self.scale = torch.sigmoid(y)
        return x * self.scale


class CBAMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(channels)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        y = x

        y = self.channel_attention(y)
        y = self.spatial_attention(y)

        return y
