from typing import Tuple

import torch
from torch import nn
from torchvision.models.resnet import resnet34

import numpy as np

from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DANN(nn.Module):
    def __init__(self) -> None:
        super(DANN, self).__init__()
        self.backbone = resnet34()
        self.backbone.fc = nn.Linear(in_features=512, out_features=2)
        self.domain_classifier = nn.Linear(in_features=512, out_features=5)

    def forward(self, x) -> Tuple[torch.Tensor]:
        for l in list(self.backbone.children())[:-1]:
            x = l(x)
        f = x.squeeze()
        f_r = ReverseLayerF.apply(f, 0.8)

        y = self.backbone.fc(f)
        d = self.domain_classifier(f_r)

        return (y, d)