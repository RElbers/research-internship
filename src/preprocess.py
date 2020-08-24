from pathlib import Path

from data.domain import Database
from preprocessing import create_patches, convert_images, clean_images, combine_masks, parse_mammograms

if __name__ == '__main__':
    csv_dir = Path(r"C:\breast_cancer")
    cbis_ddsm_dir = Path(r"C:\breast_cancer\CBIS-DDSM")
    data_dir = Path(r"C:\breast_cancer\data")

    print("[1/5] Parse mammograms")
    parse_mammograms.main(csv_dir, cbis_ddsm_dir, data_dir)
    mammograms = Database.load_mammograms(data_dir)
    database = Database(mammograms, data_dir, as_8bit=True, patch_size=1024)

    print(f"[2/6] Convert images")
    convert_images.main(cbis_ddsm_dir, database)

    print(f"[3/6] Clean images")
    clean_images.main(database)

    print(f"[4/6] Combine masks")
    combine_masks.main(database)

    database = Database(mammograms, data_dir, as_8bit=True, patch_size=512)
    print(f"[5/6] Create patches (512x512)")
    create_patches.main(database)

    database = Database(mammograms, data_dir, as_8bit=True, patch_size=1024)
    print(f"[6/6] Create patches (1024x1024)")
    create_patches.main(database)
