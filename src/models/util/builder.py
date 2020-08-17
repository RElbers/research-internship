from models.util.blocks import *


def str_to_act(act):
    if act == 'swish':
        return lambda x: x * F.sigmoid(x)
    if act == 'lrelu':
        return nn.LeakyReLU()
    if act == 'relu':
        return torch.relu
    if act == 'tanh':
        return torch.tanh
    if act == 'prelu':
        return nn.PReLU()
    if act == 'selu':
        return nn.SELU()
    if act == 'sigmoid':
        return torch.sigmoid
    if act == 'id':
        return lambda x: x
    raise Exception(f'Activation function not supported: {act}')


def str_to_norm(norm):
    if norm == 'instance':
        return nn.InstanceNorm2d
    if norm == 'batch':
        return nn.BatchNorm2d
    if norm == 'layer':
        return nn.LayerNorm
    raise Exception(f'Normalization layer not supported: {norm}')


class Builder:
    """
    ModuleBuilder contains some helper functions with default parameters already filled in.
    """

    def __init__(self,
                 kernel_size,
                 act,
                 norm,
                 antialiased_conv=False,
                 cbam=False):
        self.kernel_size = kernel_size
        self.act = act
        self.norm = norm
        self.antialiased_conv = antialiased_conv
        self.cbam = cbam

        # Call setters to resolve strings to objects
        self.set_act(act)
        self.set_norm(norm)

        # Keep old values, so we can restore them later
        self.old_kernel_size = self.kernel_size
        self.old_act = self.act
        self.old_norm = self.norm

    def down(self, input_size, filters, stride=1):
        return nn.Conv2d(in_channels=input_size,
                         out_channels=filters,
                         kernel_size=1,
                         stride=stride)

    def conv(self, input_size, filters, stride=1):
        return ConvBlock(in_channels=input_size,
                         out_channels=filters,
                         kernel_size=self.kernel_size,
                         stride=stride,
                         act=self.act,
                         norm=self.norm)

    def res(self, input_size, filters, stride=1):
        return ResidualBlock(in_channels=input_size,
                             out_channels=filters,
                             kernel_size=self.kernel_size,
                             stride=stride,
                             act=self.act,
                             norm=self.norm,
                             using_cbam=self.cbam)

    #####################################################################
    # __enter__, __exit__ and setters to temporarily change attributes. #
    #####################################################################
    def __enter__(self, *args):
        pass

    def __exit__(self, *args):
        self.kernel_size = self.old_kernel_size
        self.act = self.old_act
        self.norm = self.old_norm

    def set(self, act=-1, norm=-1, kernel_size=-1):
        if act is not -1:
            self.set_act(act)
        if norm is not -1:
            self.set_norm(norm)
        if kernel_size is not -1:
            self.set_kernel_size(kernel_size)
        return self

    def set_act(self, act):
        self.old_act = self.act
        if isinstance(act, str):
            self.act = str_to_act(act)
        else:
            self.act = act
        return self

    def set_kernel_size(self, kernel_size):
        self.old_kernel_size = self.kernel_size
        self.kernel_size = kernel_size
        return self

    def set_norm(self, norm):
        self.old_norm = self.norm
        if isinstance(norm, str):
            self.norm = str_to_norm(norm)
        else:
            self.norm = norm
        return self
