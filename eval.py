import json

import albumentations as A
import numpy as np
import streamlit as st
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from models import LightningModelWrapper, TransferedInception


def preprocess_image(img: Image.Image) -> torch.Tensor:
    transform = A.Compose(
        [
            A.Resize(299, 299),
            A.Normalize(mean=(0, 0, 0),
                        std=(1, 1, 1), always_apply=True),
            ToTensorV2()
        ]
    )
    img = np.asarray(img)
    img = transform(image=img)['image']
    return img


@st.cache
def load_class_data(path: str) -> list:
    with open(path, 'r') as file:
        file_data = file.read()
    class_data = json.loads(file_data)
    class_data = list(class_data.values())

    return class_data


def predict(img: torch.Tensor, model: LightningModelWrapper) -> torch.Tensor:
    logits = model.forward(img.unsqueeze(0))
    y_hat = torch.argmax(logits, dim=1)
    y_hat = y_hat.int().item()
    return y_hat


@st.cache
def prepare_model(model_name: str) -> LightningModelWrapper:
    torch.hub.set_dir('torchhub')
    if model_name == 'INCEPTION':
        inception_core = TransferedInception()
        model = LightningModelWrapper.load_from_checkpoint("model_parameters/inceptionv3.ckpt",
                                                           model=inception_core, learning_rate=0.01)
        model.eval()
        return model
