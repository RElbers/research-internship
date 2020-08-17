import imgaug.augmenters as iaa
from pydicom import read_file

from util.func import parallel_map


def mammogram_to_png(mammogram, database, cbis_ddsm_dir):
    img = read_file(str(cbis_ddsm_dir.joinpath(mammogram.path_img))).pixel_array
    database.paths.full(mammogram).save(img)

    resized_masks = []
    for abnormality in mammogram.abnormalities:
        crop = read_file(str(cbis_ddsm_dir.joinpath(abnormality.path_crop))).pixel_array
        database.paths.crop(abnormality).save(crop)

        mask = read_file(str(cbis_ddsm_dir.joinpath(abnormality.path_mask))).pixel_array

        # Check if mask has the same size as the full scan
        if not mask.shape == img.shape:
            print(f"\nMask is wrong shape ({img.shape}) vs ({mask.shape}):")
            print(f"\t {abnormality}")
            resized_masks.append([abnormality.id, str(img.shape), str(mask.shape)])

            resize = iaa.Resize({"width": img.shape[1], "height": img.shape[0]})
            mask = resize(images=[mask])[0]

        database.paths.mask(abnormality).save(mask)

    return resized_masks


def main(cbis_ddsm_dir, database):
    resized_masks = parallel_map(lambda mammogram: mammogram_to_png(mammogram, database=database, cbis_ddsm_dir=cbis_ddsm_dir),
                                 database.mammograms,
                                 n_threads=3)
    resized_masks = sum(list(resized_masks), [])

    path = database.paths.data_dir.joinpath("resized_masks.txt")
    with open(path, "w") as f:
        for mask in resized_masks:
            f.write(','.join(mask) + '\n')
