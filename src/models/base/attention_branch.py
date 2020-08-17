import torch.nn as nn


class AttentionBranch(nn.Module):
    def __init__(self, filters, builder):
        super().__init__()
        self.builder = builder

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
