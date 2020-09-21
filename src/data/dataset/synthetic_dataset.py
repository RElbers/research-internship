import random

import numpy as np
from PIL import Image
from PIL import ImageDraw

from data.dataset.base_dataset import BaseDataset
from data.domain import ImagePath
from util.func import parallel_map


def circle_line_triangle(draw, size, label):
    def rand_xy(x, y, r):
        r = int(r * 0.8)
        return (random.randint(x - r, x + r),
                random.randint(y - r, y + r))

    r = random.randint(max(32, size // 32), size // 8)
    x = random.randint(r, size - r)
    y = random.randint(r, size - r)

    brightness = random.randint(0, 255)
    draw.ellipse(((x - r, y - r),
                  (x + r, y + r)), outline=brightness, width=4)

    if label == 0:
        # Line
        width = random.randint(4, 32)
        draw.line((rand_xy(x, y, r),
                   rand_xy(x, y, r)), fill=(brightness), width=width)
    elif label == 1:
        # Triangle
        draw.polygon((rand_xy(x, y, r),
                      rand_xy(x, y, r),
                      rand_xy(x, y, r)), fill=brightness)

    mask = Image.fromarray(np.zeros(shape=(size, size), dtype=np.uint8))
    draw = ImageDraw.Draw(mask)
    draw.ellipse(((x - r, y - r),
                  (x + r, y + r)), fill=255)

    mask = np.array(mask)
    return draw, mask


def make_background(draw, size):
    for _ in range(32):
        brightness = random.randint(0, 255)
        if random.uniform(0, 1) > 0.5:
            # Line
            width = random.randint(4, 32)
            draw.line(((random.randint(0, size), random.randint(0, size)),
                       (random.randint(0, size), random.randint(0, size))), fill=(brightness), width=width)
        else:
            # Triangle
            draw.polygon(((random.randint(0, size), random.randint(0, size)),
                          (random.randint(0, size), random.randint(0, size)),
                          (random.randint(0, size), random.randint(0, size))), fill=brightness)

    return draw


def make(size, label):
    img = Image.fromarray(np.zeros(shape=(size, size), dtype=np.uint8))
    draw = ImageDraw.Draw(img)

    draw = make_background(draw, size)
    x, mask = circle_line_triangle(draw, size, label)
    img = np.array(img)

    # Add noise
    gauss = np.random.normal(0, 8, (size, size))
    background = img + gauss

    background = (background - background.min()) * (1 / (background.max() - background.min()) * 255)

    return np.array(background, dtype=np.uint8), np.array(mask, dtype=np.uint8)


class SyntheticDataset(BaseDataset):
    def _get_classes(self):
        return [0, 1]

    def _get_data(self):
        out_dir = self.database.data_dir.joinpath('synthetic')
        out_dir.mkdir(exist_ok=True)

        xs = []
        ys = []
        n = 2000
        size = 512

        for label in reversed(self._get_classes()):
            data = parallel_map(lambda i: SyntheticDataset.generate(out_dir, label, size, i), range(n), 4)
            x, y = zip(*data)
            xs.extend(x)
            ys.extend(y)

        return xs, ys

    @staticmethod
    def generate(out_dir, label, size, i):
        img_path = out_dir.joinpath(f"{label}_{i:04d}_img.png")
        mask_path = out_dir.joinpath(f"{label}_{i:04d}_mask.png")

        x, mask = make(size=size, label=label)
        x = np.array(x, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)

        Image.fromarray(x).save(img_path)
        Image.fromarray(mask).save(mask_path)

        xs = (ImagePath(img_path, as_8bit=True),
              ImagePath(mask_path, as_8bit=True))
        ys = label
        data = (xs, ys)
        return data
