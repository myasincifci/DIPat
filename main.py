import mlflow
from mlflow import log_metric

import random

import numpy as np
import torch
import torchvision.transforms as T
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision.models.resnet import resnet34, resnet50
from tqdm import tqdm
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader, get_train_loader
from wilds.common.grouper import CombinatorialGrouper

from model import DANN

def validate(model: nn.Module, val_loader: DataLoader, device: str) -> float:
    model.eval()
    num_samples_val = val_loader.dataset.__len__()

    with torch.no_grad():
        correct = 0

        for x, t, _ in tqdm(val_loader):
            x, t = x.to(device), t.to(device)

            y = model(x)
            correct += (t == y.argmax(axis=1)).sum().item()

        return correct/num_samples_val


def train_one_epoch(model, train_loader, grouper, optimizer, criterion, device):
    crit_crit = nn.NLLLoss()

    for i, (x, t, metadata) in enumerate(pbar := tqdm(train_loader)):
            z = grouper.metadata_to_group(metadata)

            x, t, z = x.to(device), t.to(device), z.to(device)

            optimizer.zero_grad()

            # y, d = model(x)
            # loss_clsf = criterion(y, t)
            # loss_crit = crit_crit(d, z)
            # loss = loss_clsf + loss_crit

            y = model(x)

            loss = criterion(y, t)

            loss.backward()
            
            optimizer.step()

            batch_accuracy = (y.argmax(axis=1) == t).sum().item()/len(t)
            # batch_dom_accuracy = (d.argmax(axis=1) == z).sum().item()/len(t)

            pbar.set_description(f"Cl.-acc.: {batch_accuracy:.4f}, Dom.-acc.: {batch_accuracy}")

            # log_metric("clsf_loss", loss_clsf.item(), i)
            # log_metric("crit_loss", loss_crit.item(), i)
            log_metric("train_loss", loss.item(), i)

def main():
    # random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = get_dataset(dataset="camelyon17",
                          download=True, root_dir="../../data")

    grouper = CombinatorialGrouper(dataset, ['hospital'])

    # Get the training set
    train_data = dataset.get_subset(
        "train",
        transform=T.Compose(
            [
                T.ToTensor(),
            ]
        ),
        # frac=0.1
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
        "group", train_data, batch_size=210, grouper=grouper, n_groups_per_batch=3)

    # model = DANN()
    # model.to(device)
    model = resnet50(weights=torch.load("./model_best.pth"))
    model.fc = nn.Linear(2048, 2)
    model.cuda()

    parameters = list(model.layer4.parameters()) + list(model.fc.parameters())
    optimizer = torch.optim.AdamW(params=parameters, lr=4e-4)

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=1e-3, weight_decay=1e-3)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Train loop
    for epoch in range(2):
        train_one_epoch(
             model=model,
             train_loader=train_loader,
             grouper=grouper,
             optimizer=optimizer,
             criterion=criterion,
             device=device
        )

    val_loader = get_eval_loader("standard", val_data, batch_size=3*395)

    eval_acc = validate(model, val_loader, device)
    print(f"Eval. accuracy: {eval_acc:.4f}")


if __name__ == "__main__":
    main()
