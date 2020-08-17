import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
from torch import nn

from models.util.attention import CBAMBlock


class LayerNormActWrapper(nn.Module):
    def __init__(self, layer, out_channels, act, norm):
        super().__init__()
        self.layer = layer
        self.norm = norm if norm is None else norm(out_channels)

        self.act = act

    def forward(self, x):
        y = self.layer(x)
        if self.norm is not None:
            y = self.norm(y)
        if self.act is not None:
            y = self.act(y)
        return y


class ConvBlock(LayerNormActWrapper):
    def __init__(self, in_channels, out_channels, kernel_size, stride, act, norm, antialiased_conv=False):
        padding = (kernel_size - 1) // 2

        if antialiased_conv:
            layer = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding,
                              padding_mode='reflect')
            self.downsample = Downsample(out_channels, stride=stride)
        else:
            layer = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              padding_mode='reflect')
            self.downsample = None

        super().__init__(layer, out_channels, act, norm)

    def forward(self, x):
        y = super().forward(x)

        if self.downsample is not None:
            y = self.downsample(y)
        return y


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act, norm, stride, using_cbam):
        super().__init__()
        self.using_cbam = using_cbam

        self.downscale = None
        if stride > 1:
            self.downscale = ConvBlock(in_channels, out_channels, 1, stride, None, norm)

        self.act = act
        self.conv0 = ConvBlock(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               act=act,
                               norm=norm)
        self.conv1 = ConvBlock(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=1,
                               act=None,
                               norm=norm)

        self.cbam = None
        if using_cbam:
            self.cbam = CBAMBlock(out_channels)

    def forward(self, x):
        skip = x
        if self.downscale is not None:
            skip = self.downscale(x)

        y = self.conv1(self.conv0(x))

        if self.cbam:
            y = self.cbam(y)

        out = skip + y
        out = self.act(out)
        return out


class Downsample(nn.Module):
    """
    From:
        https://github.com/adobe/antialiased-cnns/blob/master/models_lpf/__init__.py
    """

    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2),
                          int(np.ceil(1. * (filt_size - 1) / 2)),
                          int(1. * (filt_size - 1) / 2),
                          int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        if self.filt_size == 1:
            a = np.array([1., ])
        elif self.filt_size == 2:
            a = np.array([1., 1.])
        elif self.filt_size == 3:
            a = np.array([1., 2., 1.])
        elif self.filt_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif self.filt_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif self.filt_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif self.filt_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if pad_type in ['refl', 'reflect']:
        return nn.ReflectionPad2d

    if pad_type in ['repl', 'replicate']:
        return nn.ReplicationPad2d

    if pad_type == 'zero':
        return nn.ZeroPad2d

    raise ValueError('Pad type [%s] not recognized' % pad_type)
