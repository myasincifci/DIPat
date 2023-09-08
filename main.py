# from typing import Iterable, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision.models.resnet import ResNet34_Weights, resnet34, ResNet152_Weights, resnet152, resnet50, ResNet50_Weights, resnet101
from tqdm import tqdm
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader, get_train_loader
from wilds.common.grouper import CombinatorialGrouper

from model import DANN

import random


def validate(model: nn.Module, val_loader: DataLoader, device: str) -> float:
    model.eval()
    num_samples_val = val_loader.dataset.__len__()

    with torch.no_grad():
        correct = 0

        for x, y, _ in tqdm(val_loader):
            x, y = x.to(device), y.to(device)

            y_ = model(x)
            correct += (y == y_.argmax(axis=1)).sum().item()

        return correct/num_samples_val


def main():
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = get_dataset(dataset="camelyon17",
                          download=True, root_dir="../data")

    grouper = CombinatorialGrouper(dataset, ['hospital'])

    # Get the training set
    train_data = dataset.get_subset(
        "train",
        transform=T.Compose(
            [
                T.ToTensor(),
            ]
        ),
    )

    val_data = dataset.get_subset(
        "val",
        transform=T.Compose(
            [
                T.ToTensor(),
            ]
        ),
    )

    # Prepare the standard data loader
    train_loader = get_train_loader(
        "group", train_data, batch_size=3*16, grouper=grouper, n_groups_per_batch=3)

    # model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    # model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    # model.to(device)

    model = resnet101()
    model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    model.to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Train loop
    for epoch in range(2):
        i = 0
        losses = []
        for x, t, metadata in tqdm(train_loader):
            z = grouper.metadata_to_group(metadata)

            x, t = x.to(device), t.to(device)

            optimizer.zero_grad()

            y = model(x)
            loss = criterion(y, t)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            
            if i % 100 == 0:
                print(f"Avg. train-loss: {np.array(losses).mean()}")
                losses = []
                
            i += 1

    val_loader = get_eval_loader("standard", val_data, batch_size=3*128)

    eval_acc = validate(model, val_loader, device)
    print(f"Eval. accuracy: {eval_acc}")


if __name__ == "__main__":
    main()
