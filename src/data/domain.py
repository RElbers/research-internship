import enum
import pickle
from pathlib import Path

import cv2
import numpy as np
from imageio import imsave

from util import func


class Pathology(enum.Enum):
    NEGATIVE = "NEGATIVE"
    MALIGNANT = "MALIGNANT"
    BENIGN = "BENIGN"
    BENIGN_WITHOUT_CALLBACK = "BENIGN_WITHOUT_CALLBACK"

    def __str__(self):
        return self.name


class Side(enum.Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"

    def __str__(self):
        return self.name


class View(enum.Enum):
    CC = "CC"
    MLO = "MLO"

    def __str__(self):
        return self.name


class Type(enum.Enum):
    calcification = "calcification"
    mass = "mass"

    def __str__(self):
        return self.name


class Abnormality:
    def __init__(self, mammogram, csv_row):
        self.path_crop = csv_row['cropped image file path'].strip()
        self.path_mask = csv_row['ROI mask file path'].strip()
        self.num = int(csv_row['abnormality id'])
        self.pathology = Pathology(csv_row['pathology'].strip())
        self.type = Type(csv_row['abnormality type'].strip())

        self.mammogram = mammogram
        self.id = rf'{self.mammogram}_{self.num}'
        self.csv_row = csv_row

        # Set pathology of parent mammogram.
        if self.pathology == Pathology.MALIGNANT:
            mammogram.pathology = Pathology.MALIGNANT

        if self.pathology == Pathology.BENIGN:
            if not mammogram.pathology == Pathology.MALIGNANT:
                mammogram.pathology = Pathology.BENIGN

        if self.pathology == Pathology.BENIGN_WITHOUT_CALLBACK:
            if not (mammogram.pathology == Pathology.MALIGNANT or self.mammogram.pathology == Pathology.BENIGN):
                mammogram.pathology = Pathology.BENIGN_WITHOUT_CALLBACK

    def __str__(self):
        return self.id

    def __eq__(self, other):
        if isinstance(other, Abnormality):
            return self.mammogram == other.mammogram and \
                   self.num == other.num

    def __hash__(self):
        return hash(self.id)


class Mammogram:
    def __init__(self, csv_row):
        self.patient_id = csv_row['patient_id'].strip()
        self.side = Side(csv_row['left or right breast'].strip())
        self.view = View(csv_row['image view'].strip())

        self.path_img = csv_row['image file path'].strip()

        self.id = rf'{self.patient_id}_{self.side}_{self.view}'
        self.abnormalities: [Abnormality] = []
        self.pathology = Pathology.NEGATIVE

    def __str__(self):
        return self.id

    def __eq__(self, other):
        if isinstance(other, Mammogram):
            return self.patient_id == other.patient_id and \
                   self.side == other.side and \
                   self.view == other.view

        return False

    def __hash__(self):
        return hash(self.id)


class ImagePath:
    def __init__(self, path, as_8bit):
        self.path = path
        self.as_8bit = as_8bit

    def load(self):
        img = cv2.imread(str(self.path), cv2.IMREAD_ANYDEPTH)

        if self.as_8bit:
            # Scale down to 8 bit
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
        else:
            if img.dtype == np.uint8:
                img = (img * 256).astype(np.uint16)

        return img

    def save(self, img):
        if self.as_8bit:
            # Scale down to 8 bit
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        imsave(str(self.path), img)

    def __str__(self):
        return str(self.path)


class Paths:
    def __init__(self, data_dir, as_8bit, patch_size):
        self.data_dir = data_dir
        self.as_8bit = as_8bit
        bit_str = '8bit' if as_8bit else '16bit'

        self._original = data_dir.joinpath(f'original_{bit_str}')
        self._original_full = self._original.joinpath('full')
        self._original_crop = self._original.joinpath('crop')
        self._original_mask = self._original.joinpath('mask')

        self._clean = data_dir.joinpath(f'clean_{bit_str}')
        self._clean_full = self._clean.joinpath('full')
        self._clean_mask = self._clean.joinpath('mask')
        self._clean_mask_combined = self._clean.joinpath('mask_combined')

        self._patches = data_dir.joinpath(f'patches_{bit_str}_{patch_size}')
        self._patches_crop = self._patches.joinpath('crop')
        self._patches_mask = self._patches.joinpath('mask')
        self._patches_mask_negative = self._patches.joinpath('mask_negative')
        self._patches_crop_negative = self._patches.joinpath('crop_negative')

        self.make_dirs()

    def make_dirs(self):
        for k, path in vars(self).items():
            if isinstance(path, Path):
                path.mkdir(exist_ok=True)

    def full(self, mammogram):
        path = self._original_full.joinpath(mammogram.id + '.png')
        return ImagePath(path, as_8bit=self.as_8bit)

    def mask(self, abnormality):
        path = self._original_mask.joinpath(abnormality.id + '.png')
        return ImagePath(path, as_8bit=self.as_8bit)

    def crop(self, abnormality):
        path = self._original_crop.joinpath(abnormality.id + '.png')
        return ImagePath(path, as_8bit=self.as_8bit)

    def full_clean(self, mammogram):
        path = self._clean_full.joinpath(mammogram.id + '.png')
        return ImagePath(path, as_8bit=self.as_8bit)

    def mask_clean(self, abnormality):
        path = self._clean_mask.joinpath(abnormality.id + '.png')
        return ImagePath(path, as_8bit=self.as_8bit)

    def mask_combined(self, mammogram):
        path = self._clean_mask_combined.joinpath(mammogram.id + '.png')
        return ImagePath(path, as_8bit=self.as_8bit)

    def crop_clean(self, abnormality):
        path = self._patches_crop.joinpath(abnormality.id + '.png')
        return ImagePath(path, as_8bit=self.as_8bit)

    def crop_mask(self, abnormality):
        path = self._patches_mask.joinpath(abnormality.id + '.png')
        return ImagePath(path, as_8bit=self.as_8bit)

    def crop_negative(self, identifier):
        path = self._patches_crop_negative.joinpath(identifier + '.png')
        return ImagePath(path, as_8bit=self.as_8bit)

    def mask_negative(self):
        path = self._patches_mask_negative.joinpath('mask_negative.png')
        return ImagePath(path, as_8bit=self.as_8bit)


class Database:
    @staticmethod
    def save_mammograms(mammograms, directory):
        path = str(directory.joinpath("mammograms.pkl"))
        with open(path, "wb") as f:
            pickle.dump(mammograms, f)

    @staticmethod
    def load_mammograms(directory):
        path = str(directory.joinpath("mammograms.pkl"))
        with open(path, "rb") as f:
            mammograms: [Mammogram] = pickle.load(f)

        return mammograms

    def __init__(self, mammograms, data_dir, as_8bit, patch_size):
        self.mammograms: [Mammogram] = mammograms
        self.data_dir: Path = data_dir
        self.as_8bit = as_8bit
        self.patch_size = patch_size

        self.paths: Paths = Paths(data_dir, as_8bit=as_8bit, patch_size=patch_size)

    def find_mammogram(self, id):
        """
        Find a mammogram by id.
        """

        mammogram = list(filter(lambda m: m.id == id, self.mammograms))

        if len(mammogram) == 0:
            return None
        return mammogram[0]

    def find_abnormality(self, id):
        """
        Find a abnormality by id.
        """

        abnormalities = self.abnormalities(None)
        abnormality = list(filter(lambda a: a.id == id, abnormalities))

        if len(abnormality) == 0:
            return None
        return abnormality[0]

    def abnormalities(self, pathology=None):
        """
        Get a list of all abnormalities.
        """

        if pathology is None:
            return func.flatmap(lambda m: m.abnormalities, self.mammograms)

        abnormalities = self.abnormalities(None)
        abnormalities_filtered = list(filter(lambda a: a.pathology == pathology, abnormalities))
        return abnormalities_filtered

    def patches(self):
        """
        Returns all patches.
        """

        abnormalities = self.abnormalities()
        imgs = map(lambda a: self.paths.crop_clean(a), abnormalities)
        masks = map(lambda a: self.paths.crop_mask(a), abnormalities)

        xs = list(zip(imgs, masks, abnormalities))
        xs = list(filter(lambda x: x[0].path.exists() and x[1].path.exists(), xs))

        imgs, masks, abnormalities = zip(*xs)

        return imgs, masks, abnormalities

    def patches_negative(self):
        """
        Find all patches for the negative class.
        """

        imgs = list(map(lambda m: self.paths.crop_negative(f'{m.id}_0'), self.mammograms))

        masks = [self.paths._patches_mask_negative.joinpath('mask_negative.png')] * len(imgs)
        masks = list(map(lambda mask: ImagePath(mask, as_8bit=self.as_8bit), masks))

        zs = list(zip(imgs, masks))
        zs = list(filter(lambda z: z[0].path.exists() and z[1].path.exists(), zs))
        imgs, masks = zip(*zs)

        return imgs, masks
