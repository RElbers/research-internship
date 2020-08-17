import csv
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from data.domain import ImagePath
from util import vis_util


class Checkpoint():
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
            self.save_model(self.model, f"best_model.pt")

    def save_model(self, m, file_name):
        path = os.path.join(self.directory, file_name)
        torch.save(m, path)


class Logger:
    def __init__(self, name, model, dataset, frequency):
        self.frequency = frequency
        self.model = model
        self.dataset = dataset
        self.log_path = Path("../output").joinpath(name).joinpath("logs")

    def callback(self, info):
        if info.iteration == 0:
            return

        if not info.iteration % self.frequency == 0:
            return

        self.handle_split(info, 'test')
        self.handle_split(info, 'train')

    def handle_split(self, info, split):
        path = self.log_path.joinpath(split)

        mask = self.attention_mask(split)
        if mask is not None:
            ImagePath(path.joinpath('attention').joinpath(f"{info.iteration:09d}.jpg"), as_8bit=False).save(mask)

        metrics = info.metrics_train() if split == 'train' else info.metrics_test()

        vis_util.show_roc_curve(metrics, self.dataset)
        vis_util.save_plt(path.joinpath('roc_curve'), f"roc_curve_{info.iteration:09d}.png")

        cm = vis_util.show_confusion_matrix(metrics, self.dataset)
        vis_util.save_plt(path.joinpath('confusion_matrix'), f"confusion_matrix_{info.iteration:09d}.png")
        np.savetxt(path.joinpath('confusion_matrix').joinpath(f'confusion_matrix_raw_{info.iteration:09d}.txt'), cm)

        # Save metrics
        metrics_path = path.joinpath('metrics')
        metrics_path.mkdir(exist_ok=True, parents=True)
        with open(str(metrics_path.joinpath(f'metrics_{info.iteration:09d}.pkl')), 'wb') as f:
            pickle.dump(metrics, f)

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
            return vis_util.show_attention_masks(self.dataset, self.model, split=split, mask_type='guided')
        elif self.model.using_attention_blocks:
            return vis_util.show_attention_masks(self.dataset, self.model, split=split, mask_type='block')
        else:
            return None


def checkpoint(model, name, frequency):
    """
    Saves the parameters for all models to disk.
    """
    directory = os.path.join("../output", name, "checkpoints")
    if not os.path.exists(directory):
        os.makedirs(directory)

    ckp = Checkpoint(model, directory, frequency)
    return ckp.callback


def log(model, dataset, name, frequency):
    logger = Logger(name, model, dataset, frequency)
    return logger.callback
