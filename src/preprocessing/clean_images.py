import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage import measure

from util.func import parallel_map


def crop(img, borders):
    w, h = img.shape[1], img.shape[0]
    cropped_img = img[borders['top']:h - borders['bottom'], borders['left']:w - borders['right']]

    return cropped_img


def combine_borders(borders_a: dict, borders_b: dict):
    borders = {}
    for side in borders_a.keys():
        borders[side] = borders_a[side] + borders_b[side]

    return borders


def clean_mammogram(mammogram, database, background_threshold=1000):
    img = database.paths.full(mammogram).load()

    if img.dtype == np.uint8:
        background_threshold = background_threshold / 256

    w, h = img.shape[1], img.shape[0]

    mask = (img > background_threshold)
    try:
        # Find borders
        borders = find_borders(mask)

        n_background_left = (mask[:, :w // 2] == 0).sum()
        n_background_right = (mask[:, w // 2:] == 0).sum()
        if n_background_left > n_background_right:
            # Breast is on the right side
            borders['right'] = 0
        else:
            # Breast is on the left side
            borders['left'] = 0

        # Crop image and mask
        cropped_mask = crop(mask, borders)

        # Keep only largest connected component in mask
        connected_components = measure.label(cropped_mask, connectivity=2)
        largest_connected_component = np.array(connected_components == np.argmax(np.bincount(connected_components.flat)[1:]) + 1)

        # Blur mask to create margin.
        blurred_mask = gaussian_filter(largest_connected_component.astype(np.float), sigma=15) > 0.0

        # Find borders with inverted mask to remove the excess background
        borders_new = find_borders(np.invert(blurred_mask))

        # Crop mask
        final_mask = crop(blurred_mask, borders_new)
        borders_combined = combine_borders(borders, borders_new)

        # Crop image and apply mask
        img_clean = crop(img, borders_combined) * final_mask
        database.paths.full_clean(mammogram).save(img_clean)

        # Crop combined mask
        for abnormality in mammogram.abnormalities:
            mask = database.paths.mask(abnormality).load()
            mask_clean = crop(mask, borders_combined)
            database.paths.mask_clean(abnormality).save(mask_clean)

        borders['id'] = mammogram.id
        borders['err'] = ''
        return borders

    except Exception as e:
        print(f"Caught exception cleaning image ({repr(e)}):")
        print(f"\t {mammogram.id}")

        return {
            'id': mammogram.id,
            'top': -1,
            'right': -1,
            'bottom': -1,
            'left': -1,
            'err': repr(e),
        }


def find_borders(mask, background_ratio_threshold=0.1):
    w, h = mask.shape[1], mask.shape[0]

    border_left = find_border(mask.T, background_ratio_threshold)
    border_right = find_border(np.flip(mask.T, axis=0), background_ratio_threshold)
    border_top = find_border(mask, background_ratio_threshold)
    border_bottom = find_border(np.flip(mask, axis=0), background_ratio_threshold)

    border_left = 0 if border_left is None else border_left
    border_right = w if border_right is None else border_right
    border_top = 0 if border_top is None else border_top
    border_bottom = h if border_bottom is None else border_bottom

    return {
        'top': border_top,
        'right': border_right,
        'bottom': border_bottom,
        'left': border_left,
    }


def find_border(mask, background_ratio_threshold=0.1):
    for n, xs in enumerate(mask):
        background_ratio = (xs == 0).sum() / len(xs)
        if background_ratio > background_ratio_threshold:
            return n

    return None


def main(database):
    borders = parallel_map(lambda file: clean_mammogram(file, database=database),
                           database.mammograms,
                           n_threads=1)

    path = database.paths.data_dir.joinpath('borders.csv')
    df = pd.DataFrame(borders).set_index('id')
    df.to_csv(path)
