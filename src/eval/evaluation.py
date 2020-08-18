import gc
import json
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from tqdm import tqdm

from data.domain import Database, ImagePath
from data.loaders import DataLoader
from main import get_dataset, get_model
from training.trainer import TrainInfo
from util import vis_util
from util.torch_util import get_device, numpy_to_tensor
from util.vis_util import vis_attention_map, greyscale_to_heatmap


def to_calc_mass(t, p):
    # Change labels to split by mass/calcification
    t[t == 0] = 0
    t[t == 1] = 0
    t[t == 2] = 0
    t[t == 3] = 1
    t[t == 4] = 1
    t[t == 5] = 1
    t[t == 6] = 2
    p[p == 0] = 0
    p[p == 1] = 0
    p[p == 2] = 0
    p[p == 3] = 1
    p[p == 4] = 1
    p[p == 5] = 1
    p[p == 6] = 2
    return t, p, ['calc', 'mass', 'negative']


def to_malign_benign(t, p):
    # Change labels to split by malign/benign
    t[t == 0] = 0
    t[t == 1] = 1
    t[t == 2] = 2
    t[t == 3] = 0
    t[t == 4] = 1
    t[t == 5] = 2
    t[t == 6] = 3
    p[p == 0] = 0
    p[p == 1] = 1
    p[p == 2] = 2
    p[p == 3] = 0
    p[p == 4] = 1
    p[p == 5] = 2
    p[p == 6] = 3
    return t, p, ['bwc', 'malign', 'benign', 'negative']


def dice_loss(pred, target):
    smooth = 1.
    iflat = pred.flatten()
    tflat = target.flatten()

    intersection = np.sum(iflat * tflat)

    A_sum = np.sum(iflat * iflat)
    B_sum = np.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


def rescale_to_smallest(arrays):
    min_size = (10_000_000,
                10_000_000)
    for xs in arrays:
        size = xs.shape
        if size < min_size:
            min_size = size
    downsample = lambda x: cv2.resize(x, (min_size[1], min_size[0]))

    ys = []
    for xs in arrays:
        size = xs.shape[2:]
        if not size == min_size:
            y = downsample(xs)
            ys.append(y)
        else:
            ys.append(xs)

    return ys


class Evaluator:
    def _checkpoints(self):
        checkpoint_path = self.input_path.joinpath('checkpoints')
        checkpoints = sorted(checkpoint_path.glob('*.pt'))
        return checkpoints

    def _load_model(self, checkpoint):
        self.model = None
        gc.collect()

        model = torch.load(checkpoint)
        model.to(get_device())

        return model

    def _config(self):
        config_path = self.input_path.joinpath('config.json')
        with open(str(config_path), 'r') as f:
            config = json.load(f)
        print(f'Using config:')
        print(config)

        return config

    def _load_database(self):
        mammograms = Database.load_mammograms(Path(self.config['data_dir']))
        database = Database(mammograms,
                            data_dir=Path(self.config['data_dir']),
                            as_8bit=self.config['as_8bit'],
                            patch_size=self.config['patch_size'])
        return database

    def __init__(self, input_path, name):
        """
        Helper class for evaluating a model.

        :param input_path: The path which was the output path for the model. Should contain the 'config.json' file.
        :param name: The name of the experiment.
        """

        self.model = None
        self.name = name
        self.input_path = input_path
        self.output_path = input_path.parent.joinpath('eval')
        self.metrics_path = self.output_path.joinpath('metrics')
        self.output_path.mkdir(exist_ok=True, parents=True)
        self.metrics_path.mkdir(exist_ok=True, parents=True)

        self.config = self._config()
        self.checkpoints = self._checkpoints()
        self.database = self._load_database()
        self.dataset = get_dataset(database=self.database, config=self.config)

    def attention_map(self, checkpoint, type='per_abnormality'):
        """
        Evaluate the attention map.

        :param checkpoint: the checkpoint to use.
        :param type: What type of data to use. Must be one of ['per_abnormality', 'per_mammogram']
        """

        self.model = self._load_model(checkpoint)
        if not self.model.using_guided_attention and not self.model.using_attention_blocks:
            print("Model does not use attention")
            return

        # Load the data
        if type == 'per_abnormality':
            data = self.dataset.data_test
            imgs = [x[0][0] for x in data]
            masks = [x[0][1] for x in data]

            # imgs, masks = self.database.patches_negative()
            # imgs = map(lambda a: self.database.paths.crop_clean(a), self.database.abnormalities())
            # masks = map(lambda a: self.database.paths.crop_mask(a), self.database.abnormalities())

            path = self.output_path.joinpath('attention').joinpath('abnormality')
            path.mkdir(exist_ok=True, parents=True)
        elif type == 'per_mammogram':
            imgs = map(lambda m: self.database.paths.full_clean(m), self.database.mammograms)
            masks = map(lambda m: self.database.paths.mask_combined(m), self.database.mammograms)

            path = self.output_path.joinpath('attention').joinpath('mammogram')
            path.mkdir(exist_ok=True, parents=True)
        else:
            raise Exception(f"{type} must be one of ['per_abnormality', 'per_mammogram']")

        # Quantitative
        if type == 'per_abnormality':
            self._attention_mask_quantitative(imgs, masks, type=type)

        # Qualitative
        self._attention_mask(path, imgs, masks, type=type)

    def find_best_model(self):
        """
        :return: The checkpoint for the model with the best accuracy.
        """

        best_checkpoint = None
        best_accuracy = 0.0
        for checkpoint in tqdm(list(self.checkpoints)[:5]):
            self.model = self._load_model(checkpoint)
            iteration = int(str(checkpoint.name)[:9])
            tqdm.write(f"Iteration {iteration}")

            metrics = TrainInfo.calculate_metrics(self.model,
                                                  self.dataset,
                                                  batch_size=self.config['batch_size'],
                                                  split='test',
                                                  n=10 ** 9)
            accuracy = metrics['accuracy']
            if accuracy > best_accuracy:
                best_checkpoint = checkpoint
                best_accuracy = accuracy

        return best_checkpoint

    def evaluate(self, checkpoint):
        model = self._load_model(checkpoint)
        self.evaluate_model(model)

    def evaluate_model(self, model):
        self.model = model
        iteration = 0

        metrics = TrainInfo.calculate_metrics(self.model,
                                              self.dataset,
                                              batch_size=self.config['batch_size'],
                                              split='test',
                                              n=10 ** 9)

        ys_true, ys_pred, names = np.array(metrics['ys_true']), np.array(metrics['ys_pred']), self.dataset._get_class_names()
        self._calculate_accuracies(ys_true, ys_pred, names, 'all_classes')

        ys_true, ys_pred, names = to_calc_mass(ys_true, ys_pred)
        self._calculate_accuracies(ys_true, ys_pred, names, 'calc_mass')

        ys_true, ys_pred, names = to_malign_benign(ys_true, ys_pred)
        self._calculate_accuracies(ys_true, ys_pred, names, 'malign_benign')

        vis_util.plt_roc_curve(metrics, self.dataset)
        vis_util.save_plt(self.output_path, f'{self.name}_roc_curve_{iteration:09d}.png')

        vis_util.show_confusion_matrix(metrics, self.dataset)
        vis_util.save_plt(self.output_path, f'{self.name}_confusion_matrix_{iteration:09d}.png')

    def _get_attention_mask(self):
        if self.model.using_attention_blocks:
            return self.model.attention_map_block()
        elif self.model.using_guided_attention:
            return self.model.attention_mask_guided
        return None

    def _visualize_attention_mask(self, img, mask):
        self.model.infer(img)

        attention_mask = self._get_attention_mask()
        if attention_mask is None:
            return None

        visualization = vis_attention_map(img=DataLoader.normalize(img.squeeze()),
                                          mask=DataLoader.normalize(mask.squeeze()),
                                          attention_map=attention_mask)
        return visualization

    def _attention_mask_quantitative(self, imgs, masks, type):
        dice_scores = []
        bce_losses = []
        dice_scores_bin = []
        bce_losses_bin = []
        things = list(zip(imgs, masks))
        for img, mask in tqdm(things):
            if not img.path.exists():
                continue

            if not mask.path.exists():
                continue

            img = img.load()
            mask = mask.load()

            self.model.infer(img)
            mask_predict = self._get_attention_mask()
            if mask_predict is None:
                continue

            mask_predict = mask_predict.detach().cpu().numpy().squeeze()
            size = (mask_predict.shape[-1], mask_predict.shape[-2])
            mask_true = cv2.resize(mask, size).squeeze()
            mask_predict_bin = (mask_predict > 0.5).astype(np.float32)
            mask_true_bin = (mask_true > 0.5).astype(np.float32)

            dice_score = dice_loss(mask_predict, mask_true.astype(np.float32))
            bce_loss = nn.BCELoss()(numpy_to_tensor(mask_predict, torch.DoubleTensor),
                                    numpy_to_tensor(mask_true.astype(np.uint8), torch.DoubleTensor)).item()

            dice_score_bin = dice_loss(mask_predict_bin, mask_true_bin.astype(np.float32))
            bce_loss_bin = nn.BCELoss()(numpy_to_tensor(mask_predict_bin, torch.DoubleTensor),
                                        numpy_to_tensor(mask_true_bin.astype(np.uint8), torch.DoubleTensor)).item()

            dice_scores.append(dice_score)
            dice_scores_bin.append(dice_score_bin)
            bce_losses.append(bce_loss)
            bce_losses_bin.append(bce_loss_bin)

        with open(str(self.metrics_path.joinpath(f'{self.name}_metrics.txt')), 'a') as f:
            f.write(f"Type: {type}\n")
            f.write(f"\tDice: {mean(dice_scores)}\n")
            f.write(f"\tDice (bin): {mean(dice_scores_bin)}\n")
            f.write(f"\tBCE: {mean(bce_losses)}\n")
            f.write(f"\tBCE (bin): {mean(bce_losses_bin)}\n")
            f.write(f"\n")

    def _attention_mask(self, path, imgs, masks, type):
        n = 0
        vis = []

        things = list(zip(imgs, masks))
        for img, mask in tqdm(things):
            if not img.path.exists():
                continue
            if not mask.path.exists():
                continue

            visualization = self._visualize_attention_mask(img.load(), mask.load())
            if visualization is None:
                continue

            vis.append(visualization)
            if type == 'per_abnormality':
                if len(vis) < 3:
                    continue

                vis = rescale_to_smallest(vis)
                visualization = np.concatenate(vis, axis=0)
                mask_heatmap = greyscale_to_heatmap(visualization)
                out_path = ImagePath(path.joinpath(f'{self.name}_attention_abnormality_{n}.jpg'), self.database.as_8bit)
                out_path.save(mask_heatmap)
                vis = []
            else:
                mask_heatmap = greyscale_to_heatmap(visualization)
                out_path = ImagePath(path.joinpath(f'{self.name}_attention_mammogram_{n}.jpg'), self.database.as_8bit)
                out_path.save(mask_heatmap)

            n += 1
            if n >= 32:
                return

    def _calculate_accuracies(self, ys_true, ys_pred, names, type):
        cm = confusion_matrix(ys_true, ys_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        per_class_acc = list(zip(names, cm.diagonal().round(3)))
        mean_acc = accuracy_score(ys_true, ys_pred)

        with open(str(self.metrics_path.joinpath(f'{self.name}_metrics.txt')), 'a') as f:
            f.write(f"Accuracy: {type}\n")
            f.write(f"\tmean: {round(mean_acc, 3)}\n")
            for name, acc in per_class_acc:
                f.write(f"\t{name}: {acc}\n")
            f.write(f"\n")

    def test_save(self):
        imgs = self.database.paths._clean_full.glob('*')
        ws = []
        hs = []
        for img in tqdm(list(imgs)):
            img = cv2.imread(str(img), cv2.IMREAD_ANYDEPTH)
            ws.append(img.shape[0])
            hs.append(img.shape[1])
        print(mean(ws))
        print(mean(hs))
        exit(1)

        imgs, masks, abnormalities = self.database.patches()
        img_path = imgs[0]
        img = img_path.load()

        img_t = np.array([[img]])
        img_t = DataLoader.normalize(img_t)
        img_t = DataLoader.fix_dimensions(img_t)
        img_t = numpy_to_tensor(img_t, as_type=torch.FloatTensor, to_gpu=False)
        img_t = img_t

        path1 = Path('../tests/model1.pt')
        path2 = Path('../tests/model2.pt')
        model1 = get_model(n_classes=len(self.dataset.classes), config=self.config)
        y1 = model1(img_t)
        y1_infer = model1.infer(img)
        print(y1)
        print(y1_infer)
        print()

        model2 = get_model(n_classes=len(self.dataset.classes), config=self.config)
        torch.save(model1.state_dict(), path1)
        model2.load_state_dict(torch.load(path1))
        y2 = model2(img_t)
        y2_infer = model2.infer(img)
        print(y2)
        print(y2_infer)
        print()

        torch.save(model1, path2)
        model3 = torch.load(path2)
        y3 = model3(img_t)
        y3_infer = model3.infer(img)
        print(y3)
        print(y3_infer)
        print()

        a = 2
