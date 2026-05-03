from typing import Any, Iterable, List

import torch
from torch import nn
from torchvision.models import convnext_small


class Model(nn.Module):

    def __init__(self, num_cells: int = 16, weights_path: str | None = None) -> None:
        super().__init__()

        backbone = convnext_small(weights=None)
        feature_dim = backbone.classifier[2].in_features  
        backbone.classifier[2] = nn.Identity()

        self.backbone = backbone

        self.reg_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(256, 2),
        )

        self.cell_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(256, num_cells),
        )

        self.register_buffer("lat_mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("lat_std", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("lon_mean", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("lon_std", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)

        coords_norm = self.reg_head(features)
        cell_logits = self.cell_head(features)

        return {
            "coords_norm": coords_norm,
            "cell_logits": cell_logits,
        }

    def eval(self) -> "Model":
        super().eval()
        return self

    def predict(self, batch: Iterable[Any]) -> List[List[float]]:

        self.eval()

        device = next(self.parameters()).device

        batch_list = list(batch)

        if len(batch_list) == 0:
            return []

        if isinstance(batch_list[0], torch.Tensor):
            x = torch.stack(batch_list).to(device)
        else:
            raise TypeError("Expected batch to contain torch.Tensor images from preprocess.py")

        with torch.no_grad():
            outputs = self.forward(x)
            coords_norm = outputs["coords_norm"]

            lat = coords_norm[:, 0] * self.lat_std + self.lat_mean
            lon = coords_norm[:, 1] * self.lon_std + self.lon_mean

            preds = torch.stack([lat, lon], dim=1)

        return preds.detach().cpu().tolist()


def get_model() -> Model:
    return Model()