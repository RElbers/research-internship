import torch
import torch.nn as nn
from torchvision import models


class PretrainedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)

        self.model.avgpool = None
        self.model.fc = None

        self.y0 = None
        self.y1 = None
        self.y2 = None
        self.y3 = None
        self.y4 = None
        self.downsample = None

    def forward(self, x):
        x = torch.cat([x, x, x], dim=1)

        self.y0 = self.layer0(x)
        self.y1 = self.model.layer1(self.y0)
        self.y2 = self.model.layer2(self.y1)
        self.y3 = self.model.layer3(self.y2)
        self.y4 = self.model.layer4(self.y3)

        return self.y4

    def layer0(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        return x

    def combined_output(self):
        ys = []
        ys.append(self.y0)
        ys.append(self.y1)
        ys.append(self.y2)
        ys.append(self.y3)
        ys.append(self.y4)

        min_size = self.y4.shape[2:]
        if self.downsample is None:
            self.downsample = torch.nn.Upsample(size=min_size, mode='nearest')

        ys_downsampled = []
        for y in ys:
            if not y.shape[2:] == min_size:
                y_downsampled = self.downsample(y)
                ys_downsampled.append(y_downsampled)

        ys_combined = torch.cat(ys_downsampled, dim=1)
        return ys_combined


class Encoder(nn.Module):
    def __init__(self, n_channels, filters, builder):
        super().__init__()

        self.cnn_256 = []
        with builder.set_kernel_size(7):
            self.cnn_256 += [builder.res(n_channels, filters, stride=2)]

        self.cnn_256 += [builder.res(filters, filters)]
        self.cnn_256 += [builder.res(filters, filters)]
        self.cnn_256 = nn.Sequential(*self.cnn_256)
        # self.cnn_256_down = builder.down(filters, filters, 8)

        self.cnn_128 = []
        self.cnn_128 += [builder.res(filters, filters * 2, stride=2)]
        self.cnn_128 += [builder.res(filters * 2, filters * 2)]
        self.cnn_128 += [builder.res(filters * 2, filters * 2)]
        self.cnn_128 = nn.Sequential(*self.cnn_128)
        # self.cnn_128_down = builder.down(filters * 2, filters * 2, 4)

        self.cnn_64 = []
        self.cnn_64 += [builder.res(filters * 2, filters * 4, stride=2)]
        self.cnn_64 += [builder.res(filters * 4, filters * 4)]
        self.cnn_64 += [builder.res(filters * 4, filters * 4)]
        self.cnn_64 = nn.Sequential(*self.cnn_64)
        # self.cnn_64_down = builder.down(filters * 4, filters * 4, 2)

        self.cnn_32 = []
        self.cnn_32 += [builder.res(filters * 4, filters * 8, stride=2)]
        self.cnn_32 += [builder.res(filters * 8, filters * 8)]
        self.last_layer = builder.res(filters * 8, filters * 8)
        self.cnn_32 += [self.last_layer]

        self.cnn_32 = nn.Sequential(*self.cnn_32)

        self.attention_mask = None
        # self.out_filters = filters * 8 + filters * 4 + filters * 2 + filters
        self.out_filters = filters * 8

    def forward(self, img):
        y_512 = img
        y_256 = self.cnn_256(y_512)
        y_128 = self.cnn_128(y_256)
        y_64 = self.cnn_64(y_128)
        y_32 = self.cnn_32(y_64)

        y_out = y_32
        # y_out = torch.cat([self.cnn_256_down(y_256),
        #                    self.cnn_128_down(y_128),
        #                    self.cnn_64_down(y_64),
        #                    y_32], dim=1)

        if self.last_layer.using_cbam:
            self.attention_mask = self.last_layer.cbam.spatial_attention.scale

        return y_out
