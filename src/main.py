import json
import os
import sys
from datetime import datetime
from pathlib import Path

import imgaug.augmenters as iaa
from imgaug import imgaug
from torch.optim import lr_scheduler

from data.dataset.full_image_dataset import FullImageDataset
from data.dataset.patches_dataset import PatchesDataset
from data.dataset.synthetic_dataset import SyntheticDataset
from data.domain import Database
from data.loaders import ImageLoader
from models.resnet import ResNet
from training.callbacks import checkpoint, log, LearningRateScheduleWrapper
from training.trainer import Trainer


def get_augmentation(augment=True, full=False):
    """
    Returns the imgaug augmentation pipeline.

    :param augment: Returns None iff True
    :param full: Don't full mammograms by 90 degrees, as that changes the aspect ratio
    """

    if not augment:
        return None

    augs = [
        iaa.Fliplr(0.5),

        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            rotate=(-22.5, 22.5),
            shear=(-16, 16),
            mode='reflect'
        )
    ]
    if not full:
        augs.append(iaa.Rot90(imgaug.ALL, keep_size=False))

    return iaa.Sequential(augs)


def get_dataset(database, config):
    """
     Returns an instance of BaseDataset based on config['train_on'].
    """

    train_on = config['train_on']
    width = config['resize_width']
    height = config['resize_height']
    augmentation = get_augmentation(config['augment'],
                                    full=config['train_on'] == 'full')

    if train_on == 'full':
        image_loader = ImageLoader(augmentation, resize_method='scale', resize_to=(width, height))
        dataset = FullImageDataset(database=database,
                                   image_loader=image_loader)
    elif train_on == 'patch':
        image_loader = ImageLoader(augmentation, resize_method='crop', resize_to=(width, height))
        dataset = PatchesDataset(database=database,
                                 image_loader=image_loader)
    elif train_on == 'synthetic':
        image_loader = ImageLoader(augmentation, resize_method='scale', resize_to=(width, height))
        return SyntheticDataset(database=database,
                                image_loader=image_loader)
    else:
        raise ValueError("train_on must be one of ['full', 'patch', 'synthetic']")

    return dataset


def get_model(n_classes, config):
    """
     Returns an instance of BaseModel based on the config dictionary.
    """

    attention_blocks = config['attention_blocks']
    guided_attention = config['guided_attention']
    attention_loss = config['attention_loss']
    attention_weight = config['attention_weight']
    apply_attention_mask = config['apply_attention_mask']
    lr = config['lr']
    n_layers = config['n_layers']

    return ResNet(n_classes,
                  using_attention_blocks=attention_blocks,
                  using_guided_attention=guided_attention,
                  attention_loss=attention_loss,
                  attention_weight=attention_weight,
                  apply_attention_mask=apply_attention_mask,
                  lr=lr,
                  n_layers=n_layers)


def get_trainer(model, dataset, name, frequency):
    trainer = Trainer(model)

    trainer.on_iteration_end.append(checkpoint(model, name, frequency=frequency))
    trainer.on_iteration_end.append(log(model, dataset, name=name, frequency=frequency))

    lrs = lr_scheduler.ReduceLROnPlateau(model.optimizer, factor=0.1, patience=5, verbose=True)
    trainer.on_iteration_end.append(LearningRateScheduleWrapper(lrs, frequency=frequency))

    return trainer


def load_config(filename, config=None):
    config = {} if config is None else config

    with open(filename, 'r') as f:
        data = json.loads(f.read())
        for k, v in data.items():
            config[k] = v

    return config


def save_config(name, config):
    dir = Path(os.path.join("../output", name))
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    file = str(dir.joinpath('config.json'))
    with open(file, 'w') as f:
        json.dump(config, f, indent=4)

    return config


class Main:
    def __init__(self, config):
        name = datetime.now().strftime(f"%Y-%m-%d--%H-%M-%S--[{config['name']}]")
        save_config(name, config)

        mammograms = Database.load_mammograms(Path(config['data_dir']))
        database = Database(mammograms,
                            data_dir=Path(config['data_dir']),
                            as_8bit=config['as_8bit'],
                            patch_size=config['patch_size'])
        dataset = get_dataset(database=database, config=config)

        model = get_model(n_classes=len(dataset.classes), config=config)

        trainer = get_trainer(model,
                              dataset,
                              name=name,
                              frequency=config['log_frequency'])

        self.config = config
        self.database = database
        self.dataset = dataset
        self.model = model
        self.trainer = trainer
        self.name = name

    def train(self):
        self.trainer.train(self.dataset,
                           n_iterations=self.config['iterations'],
                           batch_size=self.config['batch_size'])

    @staticmethod
    def default_config():
        return {
            'name': r'default',
            'data_dir': r'C:\breast_cancer\data',
            'iterations': 2,
            'log_frequency': 2500,
            'lr': 0.0001,

            'augment': True,
            'as_8bit': False,
            'batch_size': 16,
            'train_on': 'patch',

            'patch_size': 512,
            'resize_width': 512,
            'resize_height': 512,

            'attention_blocks': False,
            'guided_attention': False,
            'attention_loss': 'dice',
            'attention_weight': 10,

            'apply_attention_mask': True,
            'n_layers': 18,
        }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        config = Main.default_config()
    else:
        path = Path(sys.argv[1])
        config = load_config(path)

    print("Using config:")
    print(f"\t{config}")

    main = Main(config)
    main.train()

    from eval.evaluation import Evaluator

    eval = Evaluator(Path(f"../output/{main.name}"), name=config['name'])
    eval.evaluate_model(main.model)
