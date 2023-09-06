from typing import Tuple

import torch
import torchvision.transforms as T
from torch import nn
from torchvision.models.resnet import ResNet34_Weights, resnet34
from tqdm import tqdm
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from wilds.common.grouper import CombinatorialGrouper

from model import DANN


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = get_dataset(dataset="camelyon17",
                          download=True, root_dir="../data")
    grouper = CombinatorialGrouper(dataset, ['hospital'])

    # Get the training set
    train_data = dataset.get_subset(
        "train",
        transform=T.Compose(
            [T.ToTensor()]
        ),
    )

    # Prepare the standard data loader
    train_loader = get_train_loader(
        "group", train_data, batch_size=3*4, grouper=grouper, n_groups_per_batch=3)

    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    model.to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Train loop
    for x, t, metadata in tqdm(train_loader):
        z = grouper.metadata_to_group(metadata)

        x, t = x.to(device), t.to(device)

        optimizer.zero_grad()

        y = model(x)
        loss = criterion(y, t)

        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()
