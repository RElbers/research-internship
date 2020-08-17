from collections import deque
from statistics import mean

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import accuracy_score, log_loss

from util.func import Lazy
from util.torch_util import get_device


def dice(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def validate_mask(model, batches, split, n=1000):
    if not model.using_guided_attention:
        return 0

    losses = []
    for iteration, (X, y_true) in enumerate(batches):
        if split == 'train':
            if iteration * len(y_true) > n:
                break

        mask = X[:, 0:1, :, :]
        model(X)

        loss = model.attention_loss(mask)
        losses.append(loss.item())

    return mean(losses)


def validate(model, batches, split, n):
    ys_true = []
    ys_pred = []

    for iteration, (X, y_true) in enumerate(batches):
        if split == 'train':
            if iteration * len(y_true) > n:
                break
        y_pred = model(X)

        ys_true.extend(y_true.detach().cpu().numpy())
        ys_pred.extend(y_pred.detach().cpu().numpy())

    return ys_true, ys_pred


class TrainInfo:
    def __init__(self, model, dataset, batch_size, iteration, total_iterations, loss, accuracy):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.iteration = iteration
        self.total_iterations = total_iterations
        self.loss = loss
        self.accuracy = accuracy

        self.metrics_train = Lazy(lambda: TrainInfo.calculate_metrics(model, dataset, batch_size, 'train'))
        self.metrics_test = Lazy(lambda: TrainInfo.calculate_metrics(model, dataset, batch_size, 'test'))

    @staticmethod
    def calculate_metrics(model, dataset, batch_size, split, n=1000):
        data = dataset.batches(split, batch_size, augment=False)
        ys_true, ys_pred_logits = validate(model, data, split=split, n=n)
        ys_pred = np.argmax(ys_pred_logits, axis=1)
        accuracy = accuracy_score(ys_true, ys_pred)
        ys_pred_proba = softmax(ys_pred_logits)
        loss = log_loss(ys_true, ys_pred_proba, labels=list(range(len(dataset.classes))))

        # data = dataset.batches(split, batch_size, augment=False)
        # attention_loss = 0  # validate_mask(model, data, split=split)

        metrics = {
            'ys_true': ys_true,
            'ys_pred': ys_pred,
            'ys_pred_logits': ys_pred_logits,
            'accuracy': accuracy,
            'loss': loss,
            'attention_loss': 0,
        }

        return metrics


class Trainer:
    def __init__(self, model):
        self.model = model.to(get_device())
        self.on_iteration_end = []
        self.running_loss = deque(maxlen=1000)
        self.running_accuracy = deque(maxlen=1000)

    def train(self, dataset, n_iterations, batch_size):
        print("Training...")

        stream = dataset.stream('train', batch_size)
        for iteration, (x, y) in enumerate(stream):
            loss, accuracy = self.model.update(x, y)

            self.running_loss.append(loss)
            self.running_accuracy.append(accuracy)

            info = TrainInfo(self.model, dataset, batch_size, iteration, n_iterations, loss, accuracy)
            self._interation_end(info)

            if iteration >= n_iterations:
                break

    def _interation_end(self, info):
        with torch.no_grad():
            # self.model.eval()

            print(f"[{info.iteration:,}/{info.total_iterations:,}] | ", end='')
            print(f"Loss {info.loss:.4f} | Accuracy {info.accuracy:.4f} | ", end='')
            print(f"Running Loss {np.mean(self.running_loss):.4f} | Running Accuracy {np.mean(self.running_accuracy):.4f} ", end='')
            print("", flush=True)

            for handler in self.on_iteration_end:
                handler(info)

            # self.model.train()
