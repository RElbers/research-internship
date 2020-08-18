import torch.nn as nn

from models.util.builder import Builder


class AttentionBranch(nn.Module):
    def __init__(self, filters):
        super().__init__()
        builder = Builder(3, 'relu', 'batch', cbam=False)

        self.conv1 = builder.res(filters, filters)
        self.conv2 = builder.res(filters, filters)
        self.conv3 = builder.res(filters, filters)
        self.conv4 = builder.res(filters, filters)

        with builder.set_act('sigmoid'):
            self.conv_out = builder.conv(filters, 1)

    def forward(self, encoding):
        mask = self.conv1(encoding)
        mask = self.conv2(mask)
        mask = self.conv3(mask)
        mask = self.conv4(mask)
        mask = self.conv_out(mask)
        return mask
