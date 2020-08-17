import numpy as np

from util.func import parallel_map


def combine_masks(mammogram, database):
    if not database.paths.full_clean(mammogram).path.exists():
        return []

    img = database.paths.full_clean(mammogram).load()

    errors = []
    combined_mask = np.zeros_like(img)
    for abnormality in mammogram.abnormalities:
        mask = database.paths.mask_clean(abnormality)
        if not mask.path.exists():
            errors.append(mask.path)
            continue

        mask = database.paths.mask_clean(abnormality).load()
        assert img.shape == mask.shape
        combined_mask += mask

    combined_mask = (combined_mask > 0).astype(np.uint16) * (2 ** 16 - 1)
    database.paths.mask_combined(mammogram).save(combined_mask)
    return errors


def main(database):
    errors = parallel_map(lambda file: combine_masks(file, database=database),
                          database.mammograms,
                          n_threads=3)

    path = database.paths.data_dir.joinpath("errors.txt")
    with open(path, "w") as f:
        for err in errors:
            f.write(f'{err}\n')
