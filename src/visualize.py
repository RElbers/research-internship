import cv2
import numpy as np
import torch
from PIL import Image

from data.domain import Pathology
from main import get_augmentation, Main
from util.torch_util import tensor_to_numpy
from util.vis_util import show_img


def checkerboard(width, height):
    w, h = 16, 16
    img = Image.new("L", (w, h))
    pixels = img.load()

    # Make pixels white where (row+col) is odd
    for i in range(w):
        for j in range(h):
            if (i + j) % 2:
                pixels[i, j] = 255

    img = cv2.resize(np.array(img), (width, height), interpolation=cv2.INTER_NEAREST)
    return np.array(img)


def show_classes(main):
    dataset = main.dataset
    w = dataset.image_loader.width
    h = dataset.image_loader.height

    for c in list(Pathology):
        grid = np.zeros(shape=(w * 2, h * 2))

        imgs = []
        for (x, y) in main.dataset.batches('test', 1):
            img = tensor_to_numpy(x[0][0])
            if y == dataset.class_to_idx[c]:
                imgs.append(img)
            if len(imgs) > 4:
                break

        grid[:w, :h] = imgs[0]
        grid[w:, :h] = imgs[1]
        grid[:w, h:] = imgs[2]
        grid[w:, h:] = imgs[3]

        show_img(grid, title=str(c))


def start():
    config = Main.default_config()
    config['name'] = '__test__'

    main = Main(config)
    aug = get_augmentation()

    # show_classes(main)
    show_augmentations(aug, main, n=4)


def show_augmentations(aug, main, n):
    # Augmentations on grid
    for i in range(1):
        cb = checkerboard(512, 512)
        cb = aug(images=[cb])[0]
        show_img(cb)

    # Show training patches.
    for i, (x, y) in enumerate(main.dataset.batches('train', 1)):
        if i > n:
            break

        img = tensor_to_numpy(x[0][0])
        mask = tensor_to_numpy(x[0][1])
        img_mask = np.concatenate([img, mask], axis=1)
        show_img(img_mask)

    # Show test patches.
    for i, (x, y) in enumerate(main.dataset.batches('test', 1)):
        if i > n:
            break

        img = tensor_to_numpy(x[0][0])
        mask = tensor_to_numpy(x[0][1])
        img_mask = np.concatenate([img, mask], axis=1)
        show_img(img_mask)


if __name__ == '__main__':
    with torch.no_grad():
        start()
