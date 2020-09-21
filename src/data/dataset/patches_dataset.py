import random
from itertools import product

from data.dataset.base_dataset import BaseDataset
from data.domain import Pathology, Type


class PatchesDataset(BaseDataset):
    @staticmethod
    def _to_class(abnormality):
        label = (str(abnormality.type), str(abnormality.pathology))
        return label

    def _get_class_names(self):
        pathologies = [
            'bwc',
            'malign',
            'benign',
        ]

        types = [
            'calc',
            'mass',
        ]

        names = []
        for i in list(product(types, pathologies)):
            names.append(f"{i[0]} {i[1]}")
        names.append('neg')
        return names

    def _get_classes(self):
        pathologies = [
            str(Pathology.BENIGN_WITHOUT_CALLBACK),
            str(Pathology.MALIGNANT),
            str(Pathology.BENIGN),
        ]

        types = [
            str(Type.calcification),
            str(Type.mass),
        ]

        classes = list(product(types, pathologies))
        classes.append(Pathology.NEGATIVE)

        return classes

    def _get_data(self):
        xs, ys = [], []

        # Negative patches
        imgs, masks = self.database.patches_negative()
        patches = list(zip(imgs, masks))
        random.shuffle(patches)
        patches = patches[:1000]

        xs.extend(patches)
        ys.extend([Pathology.NEGATIVE] * len(patches))

        # Positive patches
        imgs, masks, abnormalities = self.database.patches()
        labels = list(map(self._to_class, abnormalities))
        patches = list(zip(imgs, masks))

        xs.extend(patches)
        ys.extend(labels)

        zs = list(filter(lambda z: z[1] in self.classes, list(zip(xs, ys))))
        xs, ys = zip(*zs)

        return xs, ys
