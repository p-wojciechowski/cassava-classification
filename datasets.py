import os
from typing import Union

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations import BaseCompose, BasicTransform
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


class CassavaDataset(Dataset):
    def __init__(
        self,
        labels_file: str,
        img_dir: str,
        transform: Union[None, BasicTransform, BaseCompose] = None,
        img_size=(448, 448),
    ):
        self.img_labels = pd.read_csv(labels_file)
        self.img_dir = img_dir
        if transform:
            self.transform = transform
        else:
            self.transform = A.Compose(
                [
                    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
                    A.Resize(*img_size),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        image = Image.open(img_path)
        image = np.asarray(image)

        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image=image)["image"]
        label = torch.tensor(label).to(torch.float32)
        return image, label
