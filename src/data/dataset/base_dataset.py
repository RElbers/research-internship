import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from util.iterable import split_into_batches
from util.torch_util import numpy_to_tensor


class BaseDataset:
    """
    Base class for datasets.
    """

    def __init__(self, database, image_loader):
        self.image_loader = image_loader
        self.database = database

        self.classes = self._get_classes()
        self.class_names = self._get_class_names()
        self.class_to_idx = {c: i for (i, c) in enumerate(self.classes)}

        xs, ys = self._get_data()
        ys = list(map(lambda y: self.class_to_idx[y], ys))

        self.data_train, self.data_test = train_test_split(list(zip(xs, ys)),
                                                           train_size=0.8,
                                                           shuffle=True,
                                                           random_state=0)

    def stream(self, split, batch_size):
        """
        Generate an infinite lazy stream of batches.
        """

        while True:
            for batch in self.batches(split, batch_size):
                yield batch

    def batches(self, split, batch_size, augment=None):
        """
        Generate a stream of batches for a single epoch.
        :param split: 'train' or 'test'
        :param batch_size: the number of samples in the minibatch
        :param augment: set to true/false to override the augmentation of the split
        :return: a stream of batches
        """

        if split == 'train':
            data = self.data_train
            if augment is None:
                augment = True
        elif split == 'test':
            data = self.data_test
            if augment is None:
                augment = False
        else:
            raise ValueError("split")

        random.shuffle(data)
        for batch in split_into_batches(data, batch_size):
            x, y = zip(*batch)
            x, y = list(x), list(y)

            xs = self.image_loader.load_batch(x, augment)
            ys = np.array(y)

            x_tensor = numpy_to_tensor(xs, torch.FloatTensor)
            y_tensor = numpy_to_tensor(ys, torch.LongTensor)
            yield x_tensor, y_tensor

    def _get_class_names(self):
        """
        :return: All class names
        """
        return self._get_classes()

    def _get_classes(self):
        """
        :return: All classes
        """
        raise NotImplementedError

    def _get_data(self):
        """
        :return: All data as a pair of ([ImagePath], [class])
        """
        raise NotImplementedError
