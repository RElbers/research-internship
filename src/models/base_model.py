import numpy as np
import torch
import torch.nn as nn

from data.loaders import DataLoader
from models.util.builder import Builder
from util.torch_util import numpy_to_tensor, tensor_to_numpy


class BaseModel(nn.Module):
    def __init__(self,
                 n_classes,
                 builder: Builder):
        super().__init__()

        self.n_channels = 1
        self.n_classes = n_classes
        self.builder = builder

        self.optimizer = None
        self.attention_mask_guided = None
        self.loss = nn.CrossEntropyLoss()

    def forward(self, img):
        raise NotImplementedError

    def calculate_loss(self, img, mask, y):
        y_pred = self.forward(img)
        classification_loss = self.loss(y_pred, y)
        loss = classification_loss
        return loss, y_pred

    def update(self, xs, ys):
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
        with torch.no_grad():
            # self.eval() messes with the attention mask in the background
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

    def attention_mask_block(self):
        raise NotImplementedError()
