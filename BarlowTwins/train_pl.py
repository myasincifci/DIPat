from functools import partial
from typing import Any, Sequence, Union, Tuple
from torch import Tensor
import torch.nn.functional as F
from torchmetrics.functional import accuracy

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from lightly.loss import BarlowTwinsLoss
from lightly.models.barlowtwins import BarlowTwinsProjectionHead
from lightly.transforms.byol_transform import (BYOLTransform,
                                               BYOLView1Transform,
                                               BYOLView2Transform)
from lightly.transforms.utils import IMAGENET_NORMALIZE
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchvision import transforms as T
from torchvision.models.resnet import ResNet50_Weights, resnet50
from wilds import get_dataset


class BarlowTwins(L.LightningModule):
    def __init__(self, lr, backbone, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(2048, 2048, 2048)

        self.criterion = BarlowTwinsLoss()
        self.lr = lr

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        (x0, x1), _, _ = batch
        bs = x0.shape[0]

        z0, z1 = self.backbone(x0).view(bs, -1), self.backbone(x1).view(bs, -1)

        loss = self.criterion(z0, z1)

        wandb.log({"training-loss": loss.item()})

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        (x0, x1), _, _ = batch
        bs = x0.shape[0]

        z0, z1 = self.backbone(x0).view(bs, -1), self.backbone(x1).view(bs, -1)

        loss = self.criterion(z0, z1)

        return loss

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(params=self.parameters(), lr=self.lr)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                self.linear_warmup_decay(1000),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def fn(self, warmup_steps, step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            return 1.0

    def linear_warmup_decay(self, warmup_steps):
        return partial(self.fn, warmup_steps)
    
class OnlineFineTuner(Callback):
    def __init__(
        self,
        encoder_output_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.optimizer: torch.optim.Optimizer

        self.encoder_output_dim = encoder_output_dim
        self.num_classes = num_classes

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # add linear_eval layer and optimizer
        pl_module.online_finetuner = nn.Linear(self.encoder_output_dim, self.num_classes).to(pl_module.device)
        self.optimizer = torch.optim.Adam(pl_module.online_finetuner.parameters(), lr=1e-4)

    def extract_online_finetuning_view(
        self, batch: Sequence, device: Union[str, torch.device]
    ) -> Tuple[Tensor, Tensor]:
        (_, _, finetune_view), y = batch
        finetune_view = finetune_view.to(device)
        y = y.to(device)

        return finetune_view, y

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:
        x, y = self.extract_online_finetuning_view(batch, pl_module.device)

        with torch.no_grad():
            feats = pl_module(x)

        feats = feats.detach()
        preds = pl_module.online_finetuner(feats)
        loss = F.cross_entropy(preds, y)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        acc = accuracy(F.softmax(preds, dim=1), y, task="multiclass", num_classes=10)
        pl_module.log("online_train_acc", acc, on_step=True, on_epoch=False)
        pl_module.log("online_train_loss", loss, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
    ) -> None:
        x, y = self.extract_online_finetuning_view(batch, pl_module.device)

        with torch.no_grad():
            feats = pl_module(x)

        feats = feats.detach()
        preds = pl_module.online_finetuner(feats)
        loss = F.cross_entropy(preds, y)

        acc = accuracy(F.softmax(preds, dim=1), y, task="multiclass", num_classes=10)
        pl_module.log("online_val_acc", acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("online_val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

def main():
    L.seed_everything(42)

    dataset = get_dataset(dataset="camelyon17",
                          download=True, root_dir="/home/yasin/notebooks/data", unlabeled=False)  # TODO: change to unlabeled

    train_transform = BYOLTransform(
        view_1_transform=T.Compose([
            BYOLView1Transform(input_size=96, gaussian_blur=0.0),
        ]),
        view_2_transform=T.Compose([
            BYOLView2Transform(input_size=96, gaussian_blur=0.0),
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

    train_set_knn = dataset.get_subset(
        "train", frac=4096/len(train_set), transform=val_transform)
    val_set_knn = dataset.get_subset(
        "val", frac=1024/len(val_set), transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=216,
        shuffle=True,
        drop_last=False,
        num_workers=4,
    )

    backbone = resnet50(ResNet50_Weights.DEFAULT)
    backbone.fc = nn.Identity()

    barlow_twins = BarlowTwins(lr=1e-3, backbone=backbone)

    online_finetuner = OnlineFineTuner(2048, num_classes=2)

    trainer = L.Trainer(max_steps=1_000, accelerator="gpu")
    trainer.fit(model=barlow_twins, train_dataloaders=train_loader)


if __name__ == "__main__":
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="barlow-twins-wilds",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.02,
        "architecture": "ResNet 50",
        "dataset": "camelyon17",
        }
    )

    main()