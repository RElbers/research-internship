import numpy as np
import torch
import torch.nn as nn

from data.loaders import DataLoader
from util.torch_util import numpy_to_tensor, tensor_to_numpy


class BaseModel(nn.Module):
    """
    Base CNN class.
    """

    def __init__(self, n_classes):
        super().__init__()

        self.n_channels = 1
        self.n_classes = n_classes

        self.optimizer = None
        self.attention_map_guided = None
        self.loss = nn.CrossEntropyLoss()

    def forward(self, img):
        """
        Apply model to a minibatch
        """

        raise NotImplementedError

    def calculate_loss(self, img, mask, ys):
        """
        :param img: minibatch of images
        :param mask: minibatch of masks
        :param ys: minibatch of labels
        :return: tuple of the loss and predicted labels
        """

        raise NotImplementedError

    def update(self, xs, ys):
        """
        Update the model, given a minibatch.
        :param xs: samples
        :param ys: target labels
        :return: (loss, accuracy) for the minibatch
        """

        img = xs[:, 0:1, :, :]
        mask = xs[:, 1:2, :, :]

        loss, y_pred = self.calculate_loss(img, mask, ys)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss = loss.item()
        acc = torch.mean((ys == y_pred.argmax(dim=1)).float()).item()
        return loss, acc

    def infer(self, img):
        """
        Apply model to a single numpy image.
        :return: logits and predicted label
        """

        with torch.no_grad():
            # self.eval()

            # Minimal preprocessing
            img = np.array([img])
            img = DataLoader.normalize(img)
            img = DataLoader.fix_dimensions(img)

            img_tensor = numpy_to_tensor(img, torch.FloatTensor)
            y_logits = self(img_tensor)[0]
            y_pred = torch.argmax(y_logits)

            y_logits = tensor_to_numpy(y_logits)
            y_pred = y_pred.item()

            self.train()
        return y_logits, y_pred

    def attention_map_block(self):
        raise NotImplementedError()
