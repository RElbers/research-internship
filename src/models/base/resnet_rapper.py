import torch
import torch.nn as nn
from torch.nn import Sequential
from torchvision.models.resnet import BasicBlock, resnet18, resnet34

from models.util.attention import CBAMBlock
from util.torch_util import get_device, rescale_to_smallest


class BasicBlockWrapper(nn.Module):
    def __init__(self, block, using_cbam):
        super().__init__()
        self.block = block

        self.using_cbam = using_cbam
        self.cbam = CBAMBlock(self.block.conv2.out_channels).to(get_device())

    def forward(self, x):
        identity = x

        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        out = self.block.conv2(out)
        out = self.block.bn2(out)

        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        out += identity
        out = self.block.relu(out)

        if self.cbam:
            out = self.cbam(out)

        return out

    def attention_mask(self):
        if not self.using_cbam:
            return None

        # spatial_attention =self.cbam.spatial_attention.scale
        # channel_attention =self.cbam.channel_attention.scale
        # attention =  spatial_attention * channel_attention

        return self.cbam.spatial_attention.scale


class LayerWrapper(nn.Module):
    def wrap(self, module):
        if isinstance(module, BasicBlock):
            return BasicBlockWrapper(module, self.using_cbam)
        return module

    def __init__(self, layer, using_cbam):
        super().__init__()
        self.using_cbam = using_cbam
        self.output = None

        if self.using_cbam:
            modules = list(layer)
            modules = list(map(self.wrap, modules))
            self.layer = Sequential(*modules)
        else:
            self.layer = layer

    def __call__(self, x):
        self.output = self.layer(x)
        return self.output

    def attention_mask(self) -> list:
        if not self.using_cbam:
            return []

        masks = []
        for m in self.layer:
            if isinstance(m, BasicBlockWrapper):
                masks.append(m.attention_mask())
        return masks


class ResnetWrapper(nn.Module):
    def __init__(self, using_cbam, pretrained, n_layers):
        super().__init__()
        if n_layers == 18:
            self.resnet = resnet18(pretrained=pretrained)
        elif n_layers == 34:
            self.resnet = resnet34(pretrained=pretrained)
        else:
            raise ValueError("n_layers must be one of [18, 34]")

        self.final_filters = 512
        self.using_cbam = using_cbam

        # Remove fully connected layers
        self.resnet.avgpool = None
        self.resnet.fc = None

        # Wrap convolutional layers
        layer0 = Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
        )
        self.layer0 = LayerWrapper(layer0, using_cbam=using_cbam)
        self.layer1 = LayerWrapper(self.resnet.layer1, using_cbam=using_cbam)
        self.layer2 = LayerWrapper(self.resnet.layer2, using_cbam=using_cbam)
        self.layer3 = LayerWrapper(self.resnet.layer3, using_cbam=using_cbam)
        self.layer4 = LayerWrapper(self.resnet.layer4, using_cbam=using_cbam)
        self.layers = Sequential(
            self.layer0,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )

    def forward(self, x):
        # Pretrained resnet requires images with 3 channels
        y = torch.cat([x, x, x], dim=1)

        y = self.layers(y)
        return y

    def attention_mask(self):
        if not self.using_cbam:
            return []

        masks = [
            self.layer0.attention_mask(),
            self.layer1.attention_mask(),
            self.layer2.attention_mask(),
            self.layer3.attention_mask(),
            self.layer4.attention_mask(),
        ]

        return masks

    def combined_output(self):
        ys = [
            self.layer0.output,
            self.layer1.output,
            self.layer2.output,
            self.layer3.output,
            self.layer4.output,
        ]

        ys = rescale_to_smallest(ys)
        return ys
