import math
import random

import numpy as np

from preprocessing.clean_images import crop
from util.func import parallel_map


def find_borders(mask, margin):
    border_left = find_border(mask.T)
    border_right = find_border(np.flip(mask.T, axis=0))
    border_top = find_border(mask)
    border_bottom = find_border(np.flip(mask, axis=0))

    return {
        'top': max(0, border_top - margin),
        'right': max(0, border_right - margin),
        'bottom': max(0, border_bottom - margin),
        'left': max(0, border_left - margin),
    }


def find_border(mask):
    for n, xs in enumerate(mask):
        if (xs > 0).any():
            return n

    return None


def create_patches(mammogram, database, size):
    try:
        bad_masks = []
        img = database.paths.full_clean(mammogram).load()

        mask_combined = np.zeros_like(img)
        for abnormality in mammogram.abnormalities:
            mask = database.paths.mask_clean(abnormality).load()

            if not mask.any():
                print(f"No pixels in mask:")
                print(f"\t {abnormality}")
                continue

            if not img.shape == mask.shape:
                print(f"Mask is wrong shape ({img.shape}) vs ({mask.shape}):")
                print(f"\t {abnormality}")
                bad_masks.append(abnormality.id)
                continue

            borders = find_borders(mask, margin=size // 4)
            # borders = find_borders(mask, margin=8)
            borders = expand_to_size(mask, borders, size)
            borders = fix_negative_borders(borders)

            img_crop = crop(img, borders)
            if img_crop.shape[0] < size or img_crop.shape[1] < size:
                continue
            database.paths.crop_clean(abnormality).save(img_crop)

            mask_crop = crop(mask, borders)
            database.paths.crop_mask(abnormality).save(mask_crop)

            # show_img(img)
            # show_img(mask)
            # show_img(img_crop)
            # show_img(mask_crop)
            mask_combined += mask

        negative_mask = np.invert(mask_combined)
        negative_patches = _extract_patches(img, negative_mask, 4, size)

        for n, patch in enumerate(negative_patches):
            database.paths.crop_negative(f'{mammogram.id}_{n}').save(patch)

        return bad_masks

    except Exception as e:
        print(f"")
        print(f"Caught exception creating patches image ({repr(e)}):")
        print(f"\t {mammogram.id} \n")
        return []


def fix_negative_borders(borders):
    if borders['left'] < 0:
        negative_pixels = abs(min(0, borders['left']))
        borders['left'] += negative_pixels
        borders['right'] -= negative_pixels

    if borders['right'] < 0:
        negative_pixels = abs(min(0, borders['right']))
        borders['right'] += negative_pixels
        borders['left'] -= negative_pixels

    if borders['top'] < 0:
        negative_pixels = abs(min(0, borders['top']))
        borders['top'] += negative_pixels
        borders['bottom'] -= negative_pixels

    if borders['bottom'] < 0:
        negative_pixels = abs(min(0, borders['bottom']))
        borders['bottom'] += negative_pixels
        borders['top'] -= negative_pixels

    return borders


def expand_to_size(mask, borders, size):
    w = mask.shape[1] - borders['right'] - borders['left']
    if w < size:
        borders['left'] -= math.ceil((size - w) / 2)
        borders['right'] -= math.floor((size - w) / 2)

    h = mask.shape[0] - borders['bottom'] - borders['top']
    if h < size:
        borders['top'] -= math.ceil((size - h) / 2)
        borders['bottom'] -= math.floor((size - h) / 2)

    return borders


def _extract_patches(img, mask, n_patches, size, background_threshold=1000):
    if img.dtype == np.uint8:
        background_threshold = background_threshold / 256

    patches = []
    max_x = mask.shape[0] - size
    max_y = mask.shape[1] - size
    n = 0
    while len(patches) < n_patches:
        n += 1
        if n > 400:
            break
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        mask_patch = mask[x:x + size, y:y + size]
        img_patch = img[x:x + size, y:y + size]

        # If any of the pixels are outside of the mask, discard it
        if (mask_patch == 0).any():
            continue

        # If more then half of the pixels are black, discard the region.
        if np.sum(img_patch < background_threshold) / (size * size) > 0.5:
            continue

        if img_patch.shape[0] < size or img_patch.shape[1] < size:
            continue

        patches.append(img_patch)

    return patches


def main(database):
    bad_masks = parallel_map(lambda mammogram: create_patches(mammogram, database=database, size=database.patch_size),
                             database.mammograms,
                             n_threads=4)
    bad_masks = sum(bad_masks, [])

    t = np.uint8 if database.as_8bit else np.uint16
    mask_negative = np.zeros(shape=(database.patch_size, database.patch_size), dtype=t)
    database.paths.mask_negative().save(mask_negative)

    path = database.paths.data_dir.joinpath("bad_masks.txt")
    with open(path, "w") as f:
        for mask in bad_masks:
            f.write(mask + '\n')
