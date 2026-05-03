from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
import torch
from PIL import Image, ImageOps
from torchvision import transforms


IMAGE_SIZE = 224

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


image_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGE_MEAN,
            std=IMAGE_STD,
        ),
    ]
)


def _resolve_image_path(csv_path: Path, filename: str) -> Path:
    file_path = Path(filename)

    if file_path.is_absolute():
        return file_path

    candidate = csv_path.parent / file_path
    if candidate.exists():
        return candidate

    candidate = Path.cwd() / file_path
    if candidate.exists():
        return candidate

    return csv_path.parent / file_path


def preprocess_image(image: Union[str, Path, Image.Image]) -> torch.Tensor:
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    return image_transform(image)


def preprocess(image: Union[str, Path, Image.Image]) -> torch.Tensor:
    return preprocess_image(image)


def get_transform():
    return image_transform


def prepare_data(csv_path: str) -> Tuple[List[torch.Tensor], List[List[float]]]:
    csv_path_obj = Path(csv_path)
    df = pd.read_csv(csv_path_obj)

    X: List[torch.Tensor] = []
    y: List[List[float]] = []

    for _, row in df.iterrows():
        image_path = _resolve_image_path(csv_path_obj, str(row["file_name"]))

        X.append(preprocess_image(image_path))
        y.append([float(row["Latitude"]), float(row["Longitude"])])

    return X, y