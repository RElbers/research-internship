import numpy as np
import torch


def get_device():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    return device


def tensor_to_numpy(x):
    if isinstance(x, np.ndarray):
        return x

    y = x.detach().cpu().numpy()
    return y


def numpy_to_tensor(x, as_type=None, to_gpu=True):
    y = torch.from_numpy(x)
    if as_type is not None:
        y = y.type(as_type)
    if to_gpu:
        y = y.to(get_device())
    return y


def rescale_to_smallest(tensors):
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
