import numpy as np
import torch


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def tensor_to_numpy(x):
    """
    Convert a tensor to a numpy array.
    """

    if isinstance(x, np.ndarray):
        return x

    y = x.detach().cpu().numpy()
    return y


def numpy_to_tensor(x, as_type=None, to_gpu=True):
    """
    Convert a numpy array to a tensor.
    """

    y = torch.from_numpy(x)
    if as_type is not None:
        y = y.type(as_type)
    if to_gpu:
        y = y.to(get_device())
    return y


def rescale_to_smallest(tensors):
    """
    Resize a list of tensors to the size of the smallest one.
    """

    min_size = (10_000_000,
                10_000_000)
    for t in tensors:
        size = t.shape[2:]
        if size < min_size:
            min_size = size
    downsample = torch.nn.Upsample(size=min_size, mode='nearest')

    ys = []
    for t in tensors:
        size = t.shape[2:]
        if not size == min_size:
            y = downsample(t)
            ys.append(y)
        else:
            ys.append(t)

    return ys
