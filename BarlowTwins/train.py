import torch
import torchvision
from lightly.loss import BarlowTwinsLoss
from lightly.models.barlowtwins import BarlowTwinsProjectionHead
from lightly.transforms.byol_transform import (BYOLTransform,
                                               BYOLView1Transform,
                                               BYOLView2Transform)
from torch import nn
from torchvision import transforms as T
from torchvision.models.resnet import ResNet50_Weights, resnet50
from tqdm import tqdm
from wilds import get_dataset


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
                          download=True, root_dir="../data", unlabeled=False)  # TODO: change to unlabeled

    backbone = resnet50(ResNet50_Weights.DEFAULT)
    backbone.fc = nn.Identity()

    model = BarlowTwins(backbone=backbone)
    model.to(device)

    transform = BYOLTransform(
        view_1_transform=T.Compose([
            BYOLView1Transform(input_size=32, gaussian_blur=0.0),
        ]),
        view_2_transform=T.Compose([
            BYOLView2Transform(input_size=32, gaussian_blur=0.0),
        ])
    )

    train_set = dataset.get_subset("train", transform=transform)

    dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    criterion = BarlowTwinsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

    for (x0, x1), _, _ in tqdm(dataloader):
        optimizer.zero_grad()
        
        x0, x1 = x0.to(device), x1.to(device)
        z0, z1 = model(x0), model(x1)

        loss = criterion(z0, z1)

        print(loss.item())
        optimizer.step()

if __name__ == "__main__":
    main()
