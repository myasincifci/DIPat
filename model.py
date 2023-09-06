from typing import Tuple

import torch
from torch import nn
from torchvision.models.resnet import ResNet18_Weights, resnet18


class DANN(nn.Module):
    def __init__(self) -> None:
        super(DANN, self).__init__()
        self.backbone = resnet18()
        self.fc = nn.Linear(in_features=512, out_features=2)
        self.domain_classifier = nn.Linear(in_features=512, out_features=2)

    def forward(self, x) -> Tuple[torch.Tensor]:
        for l in list(self.backbone.children())[:-1]:
            x = l(x)

        y = self.backbone.fc(x)
        d = self.domain_classifier(x)

        return (y, d)