import torch
import torch.nn as nn

from models.base.attention_branch import AttentionBranch
from models.base.classifier import Classifier
from models.base.resnet_rapper import ResnetWrapper
from models.base_model import BaseModel
from models.util.builder import Builder
from util.torch_util import rescale_to_smallest


def dice_loss(pred, target):
    smooth = 1.
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)

    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


class ResNet(BaseModel):
    def __init__(self,
                 n_classes,
                 using_guided_attention=True,
                 using_attention_blocks=False,
                 pretrained=True,
                 attention_loss='bce',
                 attention_weight=10,
                 apply_attention_mask=10,
                 lr=0.0001,
                 n_layers=18,
                 pool='avg'):
        super().__init__(n_classes, Builder(3, 'relu', 'batch', cbam=using_attention_blocks))
        self.apply_attention_mask = apply_attention_mask
        self.attention_weight = attention_weight
        self.using_guided_attention = using_guided_attention
        self.using_attention_blocks = using_attention_blocks

        self.encoder = ResnetWrapper(using_cbam=using_attention_blocks,
                                     pretrained=pretrained,
                                     n_layers=n_layers)
        fs = self.encoder.final_filters

        self.classifier = Classifier(features_0=fs,
                                     features_1=fs,
                                     n_classes=n_classes,
                                     builder=self.builder,
                                     pool=pool)

        if using_guided_attention:
            self.attention_branch = AttentionBranch(filters=fs,
                                                    builder=self.builder)

            if attention_loss == 'dice':
                self.mask_loss = dice_loss
            elif attention_loss == 'bce':
                self.mask_loss = nn.BCELoss()
            else:
                raise ValueError(f'Unsupported loss for guided attention: {attention_loss}')

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, img):
        img = img[:, 0:1, :, :]
        encoding = self.encoder(img)

        if self.using_guided_attention:
            mask = self.attention_branch(encoding)
            self.attention_mask_guided = mask
            # if self.apply_attention_mask:
            encoding = mask * encoding

        y_pred = self.classifier(encoding)
        return y_pred

    def calculate_loss(self, img, mask, y):
        loss, y_pred = super().calculate_loss(img, mask, y)

        if self.using_guided_attention:
            mask_loss = self.attention_loss(mask)
            loss += mask_loss

        return loss, y_pred

    def attention_loss(self, mask):
        size = (self.attention_mask_guided.shape[-2], self.attention_mask_guided.shape[-1])
        downsample = torch.nn.Upsample(size=size, mode='nearest')

        mask_downsampled = downsample(mask)
        mask_loss = self.mask_loss(self.attention_mask_guided, mask_downsampled)

        return mask_loss * self.attention_weight

    def attention_mask_block(self):
        masks = []
        for ms in self.encoder.attention_mask():
            for m in ms:
                masks.append(m)

        masks_rescaled = rescale_to_smallest(masks)
        masks_rescaled = torch.cat(masks_rescaled, dim=1)
        avg_mask = masks_rescaled.mean(dim=1)
        return avg_mask.squeeze()
