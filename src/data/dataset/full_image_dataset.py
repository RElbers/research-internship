from itertools import product

from data.dataset.base_dataset import BaseDataset
from data.domain import Pathology, Type


class FullImageDataset(BaseDataset):
    @staticmethod
    def _to_class(mammogram):
        return str(mammogram.abnormalities[0].type), str(mammogram.pathology)

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
        return classes

    def _get_data(self):
        xs, ys = [], []

        for mammogram in self.database.mammograms:
            img = self.database.paths.full_clean(mammogram)
            mask = self.database.paths.mask_combined(mammogram)

            if not img.path.exists() or not mask.path.exists():
                continue

            xs.append([img, mask])
            ys.append(self._to_class(mammogram))

        zs = list(filter(lambda z: z[1] in self.classes, list(zip(xs, ys))))
        xs, ys = zip(*zs)

        return xs, ys
