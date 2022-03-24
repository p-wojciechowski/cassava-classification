import argparse
import os
import sys

import albumentations as A
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from models import CassavaDataset, LightningModelWrapper, MlpMixer, TransferedInception

parser = argparse.ArgumentParser()

parser.add_argument(
    "architecture",
    choices=["mlpmixer", "t-inception"],
    help="Architecture of neural network.",
)
parser.add_argument(
    "-d", "--dataset-dir", type=str, default="./data", help="Path to dataset directory."
)
parser.add_argument("-b", "--batch-size", type=int, default=16, help="Batch size.")
parser.add_argument(
    "-l",
    "--learning_rate",
    type=float,
    default=0.001,
    help="Learning rate for optimization.",
)
parser.add_argument(
    "--weighted-loss",
    action="store_true",
    help="Flag for weighted loss, based on class distribution.",
)
parser.add_argument(
    "-n",
    "--exp-name",
    type=str,
    default="default",
    help="Experiment name for Tensorboard logger.",
)
params = vars(parser.parse_args())

if not os.path.isdir(params["dataset_dir"]):
    print("The specified path does not exist.")
    sys.exit()

if params["exp_name"] == "default":
    params["exp_name"] = params["architecture"]

EXP_NAME = params["exp_name"]
DATASET_DIR = params["dataset_dir"]
BATCH_SIZE = params["batch_size"]
WEIGHTED_LOSS = params["weighted_loss"]
LEARNING_RATE = params["learning_rate"]

if params["architecture"] == "mlpmixer":
    IMAGE_SIZE = 448
    model = MlpMixer(
        (3, IMAGE_SIZE, IMAGE_SIZE),
        patch_size=32,
        hidden_channels=512,
        d_s=256,
        d_c=2048,
        mixer_blocks=4,
        out_class=5,
    )
elif params["architecture"] == "t-inception":
    IMAGE_SIZE = 299
    model = TransferedInception()
else:
    raise Exception("Cannot find specified network architecture.")

augmentations = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), always_apply=True),
        ToTensorV2(),
    ]
)

dataset_without_augs = CassavaDataset(
    os.path.join(DATASET_DIR, "train_smaller.csv"),
    os.path.join(DATASET_DIR, "train_images"),
)
dataset_with_augs = CassavaDataset(
    os.path.join(DATASET_DIR, "train_smaller.csv"),
    os.path.join(DATASET_DIR, "train_images"),
    augmentations,
)

train_indices, val_indices = train_test_split(
    list(range(len(dataset_without_augs))),
    test_size=0.2,
    stratify=dataset_without_augs.img_labels.iloc[:, 1],
    random_state=123,
)
train_dataset = Subset(dataset_with_augs, train_indices)
val_dataset = Subset(dataset_without_augs, val_indices)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

if WEIGHTED_LOSS:
    df = pd.read_csv(os.path.join(DATASET_DIR, "train.csv"))
    weights = (df["label"].value_counts() / df.shape[0]).sort_index()
    weights = 1.0 / weights
    weights = weights / weights.sum()
    weights = torch.Tensor(weights)
else:
    weights = torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0])

tb_logger = TensorBoardLogger("tensorboard_logs", name=EXP_NAME)
early_stopping = EarlyStopping(
    monitor="val_loss", min_delta=0.01, patience=4, verbose=True, mode="min"
)
model_checkpoint = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    dirpath="checkpoints",
    filename="mlp-mixer-loss{val_loss:.2f}",
    verbose=True,
)
trainer = pl.Trainer(
    max_epochs=6,
    logger=tb_logger,
    log_every_n_steps=5,
    callbacks=[early_stopping, model_checkpoint],
)

pl_mlp = LightningModelWrapper(model, LEARNING_RATE, weights)
trainer.fit(pl_mlp, train_dataloader, val_dataloader)
