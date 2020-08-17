import numpy as np
import pandas as pd
from pydicom import read_file
from tqdm import tqdm

from data.domain import Mammogram, Abnormality, Database


def concat_csvs(csv_dir):
    files = [
        csv_dir.joinpath("mass_case_description_test_set.csv"),
        csv_dir.joinpath("mass_case_description_train_set.csv"),
        csv_dir.joinpath("calc_case_description_test_set.csv"),
        csv_dir.joinpath("calc_case_description_train_set.csv")
    ]

    def read_csv(file):
        data = pd.read_csv(file)
        data = data.reset_index(drop=True)
        return data

    csv = list(map(read_csv, files))
    csv = pd.concat(csv, axis=0, sort=True)
    return csv


def is_mask(file):
    img = read_file(str(file))
    img = img.pixel_array.astype(float)

    result = np.isin(img, [0.0, 255.0]).all()

    return result


def parse_csvs(df_csv, cbis_ddsm_dir):
    mammograms = {}
    swaps = []
    for row in tqdm(list(df_csv.iterrows())):
        row = row[1]

        mammogram = Mammogram(row)
        if not mammogram in mammograms:
            mammograms[mammogram] = mammogram

        mammogram = mammograms[mammogram]
        abnormality = Abnormality(mammogram, row)
        mammogram.abnormalities.append(abnormality)

        # The mask file in the csv is often swapped with the cropped roi
        if not is_mask(cbis_ddsm_dir.joinpath(abnormality.path_mask)):
            abnormality.path_crop, abnormality.path_mask = abnormality.path_mask, abnormality.path_crop
            swaps.append(abnormality)

    return list(mammograms.values()), swaps


def main(csv_dir, cbis_ddsm_dir, data_dir):
    df_csv = concat_csvs(csv_dir=csv_dir)
    mammograms, swaps = parse_csvs(df_csv=df_csv, cbis_ddsm_dir=cbis_ddsm_dir)

    Database.save_mammograms(mammograms, data_dir)

    path = data_dir.joinpath("swapped.txt")
    with open(path, "w") as f:
        for a in swaps:
            f.write(a.id + '\n')
