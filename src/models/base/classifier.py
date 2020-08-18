import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, features_0, features_1, n_classes):
        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(in_features=features_0, out_features=features_1)
        self.fc2 = nn.Linear(in_features=features_1, out_features=n_classes)

    def forward(self, encoding):
        y = self.global_pool(encoding)
        y = y.view(encoding.shape[0], -1)

        y = self.fc1(y)
        y = torch.relu(y)
        y = self.fc2(y)

        return y
