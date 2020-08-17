from pathlib import Path

from data.domain import Database
from preprocessing import create_patches, clean_images


def start(cbis_ddsm_dir, data_dir, as_8bit):
    mammograms = Database.load_mammograms(data_dir)
    database = Database(mammograms, data_dir, as_8bit, patch_size=1024)

    # print(f"[2/5] Convert images {as_8bit}")
    # convert_images.main(cbis_ddsm_dir, database)
    #
    # print(f"[3/5] Clean images {as_8bit}")
    # clean_images.main(database)
    #
    # print(f"[4/5] Combine masks {as_8bit}")
    # combine_masks.main(database)

    # database = Database(mammograms, data_dir, as_8bit, 256)
    # print(f"[5/6] Create patches {as_8bit}")
    # create_patches.main(database)
    #
    # database = Database(mammograms, data_dir, as_8bit, 512)
    # print(f"[6/6] Create patches {as_8bit}")
    # create_patches.main(database)

    database = Database(mammograms, data_dir, as_8bit, 1024)
    print(f"[5/5] Create patches {as_8bit}")
    create_patches.main(database)


if __name__ == '__main__':
    csv_dir = Path(r"C:\breast_cancer")
    cbis_ddsm_dir = Path(r"C:\breast_cancer\CBIS-DDSM")
    data_dir = Path(rf"C:\breast_cancer\data")

    # print("[1/5] Parse mammograms")
    # parse_mammograms.main(csv_dir, cbis_ddsm_dir, data_dir)
    mammograms = Database.load_mammograms(data_dir)[50:]
    database = Database(mammograms, data_dir, as_8bit=True, patch_size=1024)
    # print(f"[3/5] Clean images {as_8bit}")
    clean_images.main(database)
    z = 2

    # start(cbis_ddsm_dir=cbis_ddsm_dir,
    #       data_dir=data_dir,
    #       as_8bit=True)
    #
    # start(cbis_ddsm_dir=cbis_ddsm_dir,
    #       data_dir=data_dir,
    #       as_8bit=False)
    #
