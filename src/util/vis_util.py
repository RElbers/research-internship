import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from util.torch_util import tensor_to_numpy

sns.set()


def save_plt(path, filename):
    path.mkdir(exist_ok=True, parents=True)
    plt.savefig(path.joinpath(filename))


def show_img(img, title='', save_to=None):
    w = img.shape[1] / 32
    h = img.shape[0] / 32

    size = 8
    r = w / h
    w = size * r
    h = size

    plt.close()
    fig = plt.figure(frameon=False)
    fig.suptitle(title, fontsize=20, color='white')
    fig.set_size_inches(w, h)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, aspect='auto')

    if save_to is not None:
        plt.savefig(save_to)
    plt.show()


def plt_to_array():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    img = Image.open(buf)
    return img


def greyscale_to_heatmap(image, cmap='viridis'):
    colormap = plt.get_cmap(cmap)
    out = colormap(image)
    return out[:, :, :3]


def show_roc_curve(metrics, dataset):
    n_classes = len(dataset.classes)
    ys_true = label_binarize(metrics['ys_true'], classes=list(range(n_classes)))
    ys_pred = softmax(metrics['ys_pred_logits'])

    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(ys_true[:, i], ys_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.close()
    plt.figure(figsize=(6, 6))

    for i, clazz in enumerate(dataset.class_names):
        plt.plot(fpr[i], tpr[i], lw=1, label=f'{clazz} (AUC = {roc_auc[i]:0.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")


def show_confusion_matrix(metrics, dataset):
    labels = list(map(lambda l: dataset.class_to_idx[l], dataset.classes))
    cm = confusion_matrix(metrics['ys_true'], metrics['ys_pred'], labels)

    plt.close()
    plt.figure(figsize=(8, 8))
    plt.title(f"Accuracy: {metrics['accuracy']}")
    sns.heatmap(cm / np.sum(cm),
                annot=True,
                fmt='.2%',
                xticklabels=dataset.class_names,
                yticklabels=dataset.class_names,
                cbar=False)

    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    return cm


def show_attention_masks(dataset, model, split, mask_type):
    masks = []
    xs, _ = next(dataset.stream(split, 4))
    for i in range(4):
        img = xs[i:i + 1, 0:1, :, :]
        mask = xs[i:i + 1, 1:2, :, :]

        model(img)
        if mask_type == 'block':
            attention_mask = model.attention_mask_block()
        elif mask_type == 'guided':
            attention_mask = model.attention_mask_guided
        else:
            raise ValueError("mask_type")

        mask_heatmap = show_attention_mask(img=img, mask=mask, attention_mask=attention_mask)
        masks.append(mask_heatmap)

    mask_concat = np.concatenate(masks, axis=0)
    mask_heatmap = greyscale_to_heatmap(mask_concat)
    return mask_heatmap


def show_attention_mask(img, mask, attention_mask):
    img = tensor_to_numpy(img).squeeze()
    mask = tensor_to_numpy(mask).squeeze()
    attention_mask = tensor_to_numpy(attention_mask).squeeze()

    mask_pred = cv2.resize(np.array(attention_mask), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask_concat = np.concatenate([img, mask, mask_pred], axis=1)
    mask_concat = (mask_concat * 255).astype(np.uint8)

    return mask_concat[:, :]
