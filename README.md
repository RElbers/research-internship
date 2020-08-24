# Repository for:  *Using attention for the classification of breast cancer in mammograms*.

## Usage
To use the GPU for training you also need to have CUDA and cuDNN installed. You can install the python dependencies with pip.
```bash
 $ pip install -r requirements.txt
```

To start training you just run the main file. This will use the default configuration, which is in the ```default_config()``` method in main.py.
To use the configuration from a .json file just add it as a parameter. The tests folder contains .json files for several of the experiments.
 Make sure that the ```data_dir``` is correct in the default_config function in ```main.py``` and in the .json files (see the **data** section).
```bash
 $ python3 src/main.py
 $ python3 src/main.py my_config.json
```

## Data
The CBIS-DDSM dataset is used, but first that data needs to be parsed and preprocessed before it can be used by the model.
The data can be downloaded here: https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM.
To start preprocessing, change the paths to the correct paths according to where you saved the data then run ```preprocess.py```.
```data_dir``` will contain the preprocessed images. 
During the preprocessing stage, the csv files are parsed to create a list of ```Mammogram``` objects. 
This list is saved to disk and is used to create a ```Database``` object, which provides an interface to load and save data.

```python
mammograms = Database.load_mammograms(data_dir)
database = Database(mammograms, data_dir, as_8bit=True, patch_size=1024)

# Positive patches
imgs, masks, abnormalities = database.patches()

# Negative patches
imgs, masks = database.patches_negative()

# All abnormalities
abnormalities = database.abnormalities()

# The paths attribute has methods getting the ImagePath of different image types.
img_path = database.paths.mask_clean(abnormalities[0])

# The ImagePath class contains methods for loading and saving.
mask = img_path.load()
```
