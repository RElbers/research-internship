import csv
import os
import pickle

import numpy as np
import torch

from data.domain import ImagePath
from util import vis_util


class Checkpoint:
    """
    Callback which saves the model every n iterations.
    """

    def __init__(self, model, directory, frequency):
        self.model = model
        self.directory = directory
        self.frequency = frequency
        self.best = 0

    def callback(self, info):
        if info.iteration == 0:
            return

        if not info.iteration % self.frequency == 0:
            return

        metrics = info.metrics_test()
        accuracy = metrics['accuracy']
        if accuracy > self.best:
            self.best = accuracy
            self.save_model(self.model, f"best_model_{info.iteration}.pt")

        self.save_model(self.model, f"model_{info.iteration}.pt")

    def save_model(self, m, file_name):
        path = os.path.join(self.directory, file_name)
        torch.save(m, path)


class Logger:
    """
    Callback which logs some metrics of the model every n iterations.
    """

    def __init__(self, path, model, dataset, frequency):
        self.frequency = frequency
        self.model = model
        self.dataset = dataset
        self.log_path = path

    def callback(self, info):
        if info.iteration == 0:
            return

        if not info.iteration % self.frequency == 0:
            return

        self.handle_split(info, 'test')
        self.handle_split(info, 'train')

    def handle_split(self, info, split):
        path = self.log_path.joinpath(split)

        # Log attention mask
        mask = self.attention_mask(split)
        if mask is not None:
            ImagePath(path.joinpath('attention').joinpath(f"{info.iteration:09d}.jpg"), as_8bit=False).save(mask)

        # Calculate metrics
        metrics = info.metrics_train() if split == 'train' else info.metrics_test()

        # ROC curve
        vis_util.plt_roc_curve(metrics, self.dataset)
        vis_util.save_plt(path.joinpath('roc_curve'), f"roc_curve_{info.iteration:09d}.png")

        # Confusion matrix
        cm = vis_util.show_confusion_matrix(metrics, self.dataset)
        vis_util.save_plt(path.joinpath('confusion_matrix'), f"confusion_matrix_{info.iteration:09d}.png")
        np.savetxt(path.joinpath('confusion_matrix').joinpath(f'confusion_matrix_raw_{info.iteration:09d}.txt'), cm)

        # Save the raw metrics
        metrics_path = path.joinpath('metrics')
        metrics_path.mkdir(exist_ok=True, parents=True)
        with open(str(metrics_path.joinpath(f'metrics_{info.iteration:09d}.pkl')), 'wb') as f:
            pickle.dump(metrics, f)

        #  Save just the accuracy and losses
        with open(metrics_path.joinpath('data.csv'), "a") as f:
            header = ['accuracy', 'loss', 'attention_loss']
            m = {
                'accuracy': metrics['accuracy'],
                'loss': metrics['loss'],
                'attention_loss': metrics['attention_loss']
            }

            writer = csv.DictWriter(f, fieldnames=header)
            writer.writerow(m)

    def attention_mask(self, split):
        if self.model.using_guided_attention:
            return vis_util.vis_attention_maps(self.dataset, self.model, split=split, mask_type='guided')
        elif self.model.using_attention_blocks:
            return vis_util.vis_attention_maps(self.dataset, self.model, split=split, mask_type='block')
        else:
            return None


class LearningRateScheduleWrapper:
    """
    Callback for decaying the learning rate.
    """

    def __init__(self, lr_scheduler, frequency=1):
        super().__init__()
        self.frequency = frequency
        self.scheduler = lr_scheduler
        self.accuracy = None

    def __call__(self, info):
        if info.iteration == 0:
            return

        if not info.iteration % self.frequency == 0:
            return

        self.accuracy = info.metrics_test()['accuracy']
        return self.update(info.iteration)

    def update(self, idx):
        return self.scheduler.step(self.accuracy)


def checkpoint(model, name, frequency):
    directory = os.path.join("../output", name, "checkpoints")
    if not os.path.exists(directory):
        os.makedirs(directory)

    ckp = Checkpoint(model, directory, frequency)
    return ckp.callback


def log(model, dataset, name, frequency):
    logger = Logger(name, model, dataset, frequency)
    return logger.callback
