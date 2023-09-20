import mlflow
import torch
import torchvision
from lightly.loss import BarlowTwinsLoss
from lightly.models.barlowtwins import BarlowTwinsProjectionHead
from lightly.transforms.byol_transform import (BYOLTransform,
                                               BYOLView1Transform,
                                               BYOLView2Transform)
from lightly.transforms.utils import IMAGENET_NORMALIZE
from sklearn.neighbors import KNeighborsClassifier
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms as T
from torchvision.models.resnet import ResNet50_Weights, resnet50
from tqdm import tqdm
from wilds import get_dataset

def knn_online_eval(model: nn.Module, train_set, val_set, device):
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=100,
        shuffle=True,
        num_workers=4,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=100,
        shuffle=True,
        num_workers=4,
    )

    with torch.no_grad():
        # Compute train embeddings
        Z, T = [], []
        for x, t, _ in tqdm(train_loader, desc="Train knn"):
            z = model(x.to(device)).cpu().detach()
            t = t.cpu().detach()
            Z.append(z)
            T.append(t)
        Z, T = torch.cat(Z, dim=0), torch.cat(T, dim=0)

        neigh = KNeighborsClassifier(n_neighbors=2)
        neigh.fit(Z, T)

        correct = 0
        for x, t, _ in tqdm(val_loader, desc="Eval. knn"):
            z = model(x.to(device)).cpu().detach()
            t = t.cpu().detach().numpy()
            y = neigh.predict(z)
            correct += (y == t).sum()

    return correct / len(val_set)


class BarlowTwins(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(2048, 2048, 2048)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = get_dataset(dataset="camelyon17",
                          download=True, root_dir="/home/yasin/notebooks/data", unlabeled=False)  # TODO: change to unlabeled

    backbone = resnet50(ResNet50_Weights.DEFAULT)
    backbone.fc = nn.Identity()

    model = BarlowTwins(backbone=backbone)
    model.to(device)

    tf_args = {
        "input_size": 96,
        "gaussian_blur": 0.0,
        "min_scale": 0.1
    }
    train_transform = BYOLTransform(
        view_1_transform=T.Compose([
            BYOLView1Transform(input_size=96, gaussian_blur=0.0, min_scale=0.1, solarization_prob=0.0),
        ]),
        view_2_transform=T.Compose([
            BYOLView2Transform(input_size=96, gaussian_blur=0.0, min_scale=0.1, solarization_prob=0.0),
        ])
    )

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
                mean=IMAGENET_NORMALIZE["mean"],
                std=IMAGENET_NORMALIZE["std"],
        ),
    ])

    train_set = dataset.get_subset("train", transform=train_transform)
    val_set = dataset.get_subset("val", transform=train_transform)

    train_data_knn = dataset.get_subset(
        "train", frac=4096/len(train_set), transform=val_transform)
    val_data_knn = dataset.get_subset(
        "val", frac=1024/len(val_set), transform=val_transform)

    dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=210,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    criterion = BarlowTwinsLoss(lambda_param=0.005)
    optimizer = optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.05)

    accs = [0]
    for epoch in range(10):
        for i, ((x0, x1), _, _) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()

            x0, x1 = x0.to(device), x1.to(device)
            z0, z1 = model(x0), model(x1)

            loss = criterion(z0, z1)
            loss.backward()

            print(loss.item())
            mlflow.log_metric("bt-loss", loss.item(),
                              epoch*len(dataloader) + i)

            optimizer.step()

            if i % 100 == 0:
                val_acc = knn_online_eval(
                    model, train_data_knn, val_data_knn, device)
                mlflow.log_metric("val-acc", val_acc,
                                  epoch*len(dataloader) + i)
                
                if val_acc >= max(accs):
                    print(f"New best: {val_acc}")
                    torch.save(model.backbone.state_dict(), "model_best.pth")

                accs.append(val_acc)
                


if __name__ == "__main__":
    main()
