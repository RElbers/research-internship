import cv2
import imgaug.augmenters as iaa
import numpy as np


class DataLoader:
    """
    Base class for loading images.
    """

    def __init__(self, augmentation, resize_method, resize_to):
        """

        :param augmentation:
        :param resize_method:
        :param resize_to:
        """
        if resize_method.lower() == 'crop':
            resize = self._crop_resize
        elif resize_method.lower() == 'scale':
            resize = self._scale_resize
        else:
            raise ValueError("resize_method")

        self.width = resize_to[0]
        self.height = resize_to[1]
        self.resize = resize
        self.augmentation = augmentation

    def _load(self, imgs, masks):
        raise NotImplementedError()

    def load_batch(self, x, augment):
        imgs = self._load(imgs=[i[0] for i in x],
                          masks=[i[1] for i in x])

        imgs = self._resize(imgs)
        imgs = self._augment(imgs, augment)
        imgs = DataLoader.normalize(imgs)
        imgs = DataLoader.fix_dimensions(imgs)

        return np.array(imgs)

    def _resize(self, imgs):
        imgs = list(map(self.resize, imgs))
        imgs = np.array(imgs)

        return imgs

    def _augment(self, imgs, augment):
        if augment and self.augmentation is not None:
            imgs = self.augmentation(images=imgs)
            imgs = np.array(imgs)

        return imgs

    def _crop_resize(self, img):
        aug = iaa.Sequential([iaa.PadToFixedSize(width=self.width, height=self.height),
                              iaa.CropToFixedSize(width=self.width, height=self.height)])
        return aug(images=[img])[0]

    def _scale_resize(self, img):
        return cv2.resize(img, (self.width, self.height))

    @staticmethod
    def normalize(imgs):
        # Rescale to 0-1
        if imgs.dtype == np.uint16:
            imgs = (imgs / (2 ** 16 - 1))
        else:
            imgs = imgs / 255.

        return imgs

    @staticmethod
    def fix_dimensions(imgs):
        if len(imgs.shape) < 4:
            imgs = np.expand_dims(imgs, axis=1)
        else:
            imgs = np.einsum('nijk->nkij', imgs)

        return imgs


class ImageLoader(DataLoader):
    """
    Subclass of DataLoader, for which imgs and masks are lists of ImagePaths.
    """

    def _load(self, imgs, masks):
        imgs = list(map(lambda img: img.load(), imgs))
        masks = list(map(lambda mask: mask.load(), masks))

        img_mask = list(map(lambda img_mask: np.stack(img_mask, axis=2), zip(imgs, masks)))
        return img_mask
