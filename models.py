from typing import Tuple

import pytorch_lightning as pl
import torch
import torchmetrics
import torchvision
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F


class FullPerPatch(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, hidden_channels: int):
        super().__init__()
        self.per_patch = nn.Conv2d(in_channels, hidden_channels, patch_size, patch_size)
        self.rearrange = Rearrange("b c s1 s2 -> b (s1 s2) c")

    def forward(self, x):
        x = self.per_patch(x)
        x = self.rearrange(x)
        return x


class MlpBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, channels: int, patches: int, d_s: int, d_c: int):
        super(MixerBlock, self).__init__()
        self.patch_layer_norm = nn.LayerNorm([patches, channels])
        self.mlp1 = MlpBlock(patches, d_s)
        self.mlp2 = MlpBlock(channels, d_c)
        self.channel_layer_norm = nn.LayerNorm([patches, channels])

    def forward(self, x):
        skip1 = x
        x = self.patch_layer_norm(x)
        x = torch.transpose(x, 1, 2)
        x = self.mlp1(x)
        x = torch.transpose(x, 1, 2)
        x += skip1
        skip2 = x
        x = self.channel_layer_norm(x)
        x = self.mlp2(x)
        x += skip2

        return x


class MlpMixer(nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int, int],
        patch_size: int,
        hidden_channels: int,
        d_s: int,
        d_c: int,
        mixer_blocks: int,
        out_class: int,
    ):
        super().__init__()

        assert (
            image_size[1] * image_size[2] % patch_size * patch_size == 0
        ), "Wrong size of image and patch!"

        patches = int(image_size[1] * image_size[2] // (patch_size**2))

        self.patching = FullPerPatch(
            patch_size=patch_size,
            in_channels=image_size[0],
            hidden_channels=hidden_channels,
        )
        self.mixer_blocks = nn.Sequential(
            *[
                MixerBlock(hidden_channels, patches, d_s, d_c)
                for i in range(mixer_blocks)
            ]
        )
        self.head = nn.Linear(hidden_channels, out_class)

    def forward(self, x):
        x = self.patching(x)
        x = self.mixer_blocks(x)
        x = torch.mean(x, 1)
        x = self.head(x)
        return x


class LightningModelWrapper(pl.LightningModule):
    def __init__(self, model, learning_rate, class_weights=None):
        super().__init__()
        self.model = model
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.learning_rate = learning_rate
        if class_weights is None:
            self.class_weights = torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        self.class_weights = class_weights

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y, self.class_weights)
        y_hat = torch.argmax(logits, dim=1)
        self.train_acc(y_hat, y)
        return loss

    def training_epoch_end(self, outputs):
        tensorboard = self.logger.experiment
        epoch_avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        tensorboard.add_scalars("Loss", {"train": epoch_avg_loss}, self.current_epoch)
        tensorboard.add_scalars(
            "Accuracy", {"train": self.train_acc.compute()}, self.current_epoch
        )
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y, self.class_weights)
        y_hat = torch.argmax(logits, dim=1)
        self.val_acc(y_hat, y)
        return loss

    def validation_epoch_end(self, outputs):
        tensorboard = self.logger.experiment
        epoch_avg_loss = torch.stack(outputs).mean()

        tensorboard.add_scalars("Loss", {"val": epoch_avg_loss}, self.current_epoch)
        tensorboard.add_scalars(
            "Accuracy", {"val": self.val_acc.compute()}, self.current_epoch
        )
        self.log("val_loss", epoch_avg_loss, logger=False)
        self.val_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class TransferredInception(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception = torch.hub.load(
            "pytorch/vision:v0.10.0", "inception_v3", pretrained=True
        )
        for param in self.inception.parameters():
            param.requires_grad = False
        self.inception.fc = nn.Identity()
        self.classification_head = nn.Linear(in_features=2048, out_features=5)

    def forward(self, x):
        x = self.inception.forward(x)
        if isinstance(x, torchvision.models.inception.InceptionOutputs):
            x = x[0]
        x = self.classification_head(x)
        return x
