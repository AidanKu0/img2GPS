from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import convnext_small, ConvNeXt_Small_Weights

from model import Model

FILENAME_COL = "file_name"
LAT_COL = "Latitude"
LON_COL = "Longitude"

COORD_LOSS_WEIGHT = 1.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    h = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )

    return 2 * radius * math.asin(math.sqrt(h))

def make_cell_edges(df: pd.DataFrame, grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
    lat_min, lat_max = df[LAT_COL].min(), df[LAT_COL].max()
    lon_min, lon_max = df[LON_COL].min(), df[LON_COL].max()

    if lat_min == lat_max:
        lat_min -= 1e-6
        lat_max += 1e-6

    if lon_min == lon_max:
        lon_min -= 1e-6
        lon_max += 1e-6

    lat_edges = np.linspace(lat_min, lat_max, grid_size + 1)
    lon_edges = np.linspace(lon_min, lon_max, grid_size + 1)

    return lat_edges, lon_edges


def assign_cell_id(
    lat: float,
    lon: float,
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
    grid_size: int,
) -> int:
    lat_bin = np.digitize(lat, lat_edges[1:-1], right=False)
    lon_bin = np.digitize(lon, lon_edges[1:-1], right=False)

    lat_bin = int(np.clip(lat_bin, 0, grid_size - 1))
    lon_bin = int(np.clip(lon_bin, 0, grid_size - 1))

    return lat_bin * grid_size + lon_bin


def make_stratification_labels(df: pd.DataFrame, grid_size: int) -> np.ndarray:
    lat_bins = pd.qcut(df[LAT_COL], q=grid_size, labels=False, duplicates="drop")
    lon_bins = pd.qcut(df[LON_COL], q=grid_size, labels=False, duplicates="drop")

    labels = lat_bins.astype(str) + "_" + lon_bins.astype(str)
    return labels.to_numpy()

class GPSDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform,
        grid_size: int,
        lat_mean: float | None = None,
        lat_std: float | None = None,
        lon_mean: float | None = None,
        lon_std: float | None = None,
        lat_edges: np.ndarray | None = None,
        lon_edges: np.ndarray | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.grid_size = grid_size

        self.lat_mean = float(self.df[LAT_COL].mean()) if lat_mean is None else float(lat_mean)
        self.lat_std = float(self.df[LAT_COL].std()) if lat_std is None else float(lat_std)

        self.lon_mean = float(self.df[LON_COL].mean()) if lon_mean is None else float(lon_mean)
        self.lon_std = float(self.df[LON_COL].std()) if lon_std is None else float(lon_std)

        if self.lat_std == 0:
            self.lat_std = 1e-8
        if self.lon_std == 0:
            self.lon_std = 1e-8

        if lat_edges is None or lon_edges is None:
            self.lat_edges, self.lon_edges = make_cell_edges(self.df, grid_size)
        else:
            self.lat_edges = lat_edges
            self.lon_edges = lon_edges

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"])
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = self.transform(image)

        lat = float(row[LAT_COL])
        lon = float(row[LON_COL])

        lat_norm = (lat - self.lat_mean) / self.lat_std
        lon_norm = (lon - self.lon_mean) / self.lon_std

        coords_norm = torch.tensor([lat_norm, lon_norm], dtype=torch.float32)

        cell_id = assign_cell_id(
            lat,
            lon,
            self.lat_edges,
            self.lon_edges,
            grid_size=self.grid_size,
        )

        cell_id = torch.tensor(cell_id, dtype=torch.long)

        return image, coords_norm, cell_id

def initialize_pretrained_backbone(model: Model) -> None:
    pretrained = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
    pretrained_sd = pretrained.state_dict()

    pretrained_sd = {
        k: v for k, v in pretrained_sd.items()
        if not k.startswith("classifier.")
    }

    missing, unexpected = model.backbone.load_state_dict(pretrained_sd, strict=False)

    print("Loaded pretrained ConvNeXt-Small backbone.")
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)


def set_model_normalization(
    model: Model,
    lat_mean: float,
    lat_std: float,
    lon_mean: float,
    lon_std: float,
) -> None:
    model.lat_mean.fill_(float(lat_mean))
    model.lat_std.fill_(float(lat_std))
    model.lon_mean.fill_(float(lon_mean))
    model.lon_std.fill_(float(lon_std))


def make_optimizer(
    model: Model,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )

def train_one_epoch(
    model: Model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    coord_criterion,
    cell_criterion,
    device: torch.device,
    cell_loss_weight: float,
) -> float:
    model.train()

    total_loss = 0.0

    for images, coords_norm, cell_ids in loader:
        images = images.to(device, non_blocking=True)
        coords_norm = coords_norm.to(device, non_blocking=True)
        cell_ids = cell_ids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        coord_loss = coord_criterion(outputs["coords_norm"], coords_norm)
        cell_loss = cell_criterion(outputs["cell_logits"], cell_ids)

        loss = COORD_LOSS_WEIGHT * coord_loss + cell_loss_weight * cell_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate(
    model: Model,
    loader: DataLoader,
    device: torch.device,
    lat_mean: float,
    lat_std: float,
    lon_mean: float,
    lon_std: float,
) -> Dict[str, float | np.ndarray]:
    model.eval()

    distances: List[float] = []
    preds_all: List[List[float]] = []
    actuals_all: List[List[float]] = []

    with torch.no_grad():
        for images, coords_norm, _ in loader:
            images = images.to(device, non_blocking=True)
            coords_norm = coords_norm.to(device, non_blocking=True)

            outputs = model(images)

            pred_norm = outputs["coords_norm"].detach().cpu().numpy()
            actual_norm = coords_norm.detach().cpu().numpy()

            preds = pred_norm * np.array([lat_std, lon_std]) + np.array([lat_mean, lon_mean])
            actuals = actual_norm * np.array([lat_std, lon_std]) + np.array([lat_mean, lon_mean])

            for pred, actual in zip(preds, actuals):
                pred_lat, pred_lon = float(pred[0]), float(pred[1])
                actual_lat, actual_lon = float(actual[0]), float(actual[1])

                distance = haversine_m(actual_lat, actual_lon, pred_lat, pred_lon)

                distances.append(distance)
                preds_all.append([pred_lat, pred_lon])
                actuals_all.append([actual_lat, actual_lon])

    distances_np = np.array(distances)

    return {
        "rmse_m": float(np.sqrt(np.mean(distances_np ** 2))),
        "avg_m": float(np.mean(distances_np)),
        "median_m": float(np.median(distances_np)),
        "distances": distances_np,
        "preds": np.array(preds_all),
        "actuals": np.array(actuals_all),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="/content/data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=7)       
    parser.add_argument("--val_size", type=float, default=0.20)
    parser.add_argument("--out", type=str, default="model.pt")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--grid_size", type=int, default=4)
    parser.add_argument("--cell_loss_weight", type=float, default=0.3) 

    parser.add_argument("--backbone_lr", type=float, default=1e-5)  
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3) 

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--use_all_data",
        action="store_true",
        help="Combine train+test",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    train_csv = data_dir / "train" / "metadata.csv"
    test_csv = data_dir / "test" / "metadata.csv"

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_df["image_path"] = train_df[FILENAME_COL].apply(
        lambda x: str(data_dir / "train" / str(x))
    )
    test_df["image_path"] = test_df[FILENAME_COL].apply(
        lambda x: str(data_dir / "test" / str(x))
    )

    if args.use_all_data:
        print("\n*** use_all_data=True: combining train + test***")
        all_df = pd.concat([train_df, test_df], ignore_index=True)
        train_df = all_df
        print(f"Total examples after merge: {len(train_df)}")

    print("Train examples:", len(train_df))
    if not args.use_all_data:
        print("Test examples:", len(test_df))
    print("Example train image:", train_df.loc[0, "image_path"])
    print("Exists?", Path(train_df.loc[0, "image_path"]).exists())

    if not Path(train_df.loc[0, "image_path"]).exists():
        raise FileNotFoundError(
            "Example training image path does not exist. Check data_dir and metadata filenames."
        )

    num_cells = args.grid_size * args.grid_size

    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.3,
                hue=0.08,
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    inference_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    pin_memory = device.type == "cuda"

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }

    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2


    if args.use_all_data:
        print("\nTRAINING ON ALL DATA (no val split)")

        lat_edges, lon_edges = make_cell_edges(train_df, args.grid_size)

        train_dataset = GPSDataset(
            train_df,
            transform=train_transform,
            grid_size=args.grid_size,
            lat_edges=lat_edges,
            lon_edges=lon_edges,
        )

        lat_mean = train_dataset.lat_mean
        lat_std = train_dataset.lat_std
        lon_mean = train_dataset.lon_mean
        lon_std = train_dataset.lon_std

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **loader_kwargs,
        )

        model = Model(num_cells=num_cells)
        initialize_pretrained_backbone(model)
        set_model_normalization(model, lat_mean, lat_std, lon_mean, lon_std)
        model = model.to(device)

        coord_criterion = nn.SmoothL1Loss()
        cell_criterion = nn.CrossEntropyLoss()

        optimizer = make_optimizer(
            model,
            backbone_lr=args.backbone_lr,
            head_lr=args.head_lr,
            weight_decay=args.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )

        print("\n========== TRAINING ==========")
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                coord_criterion,
                cell_criterion,
                device,
                cell_loss_weight=args.cell_loss_weight,
            )
            scheduler.step()
            print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f}")

        torch.save(model.state_dict(), args.out)
        print(f"\nSaved final model to: {args.out}")
        return

    print("\nSINGLE TRAIN/VAL SPLIT")

    strat_labels = make_stratification_labels(train_df, args.grid_size)
    label_counts = pd.Series(strat_labels).value_counts()

    if len(label_counts) > 0 and label_counts.min() >= 2:
        train_sub_df, val_sub_df = train_test_split(
            train_df,
            test_size=args.val_size,
            random_state=42,
            shuffle=True,
            stratify=strat_labels,
        )
        print("Using spatially stratified train/val split.")
    else:
        train_sub_df, val_sub_df = train_test_split(
            train_df,
            test_size=args.val_size,
            random_state=42,
            shuffle=True,
        )
        print("Using normal train/val split because some spatial bins were too small.")

    train_sub_df = train_sub_df.reset_index(drop=True)
    val_sub_df = val_sub_df.reset_index(drop=True)

    print("train_sub:", len(train_sub_df))
    print("val_sub:", len(val_sub_df))
    print("test:", len(test_df))

    lat_edges, lon_edges = make_cell_edges(train_sub_df, args.grid_size)

    train_dataset = GPSDataset(
        train_sub_df,
        transform=train_transform,
        grid_size=args.grid_size,
        lat_edges=lat_edges,
        lon_edges=lon_edges,
    )

    lat_mean = train_dataset.lat_mean
    lat_std = train_dataset.lat_std
    lon_mean = train_dataset.lon_mean
    lon_std = train_dataset.lon_std

    val_dataset = GPSDataset(
        val_sub_df,
        transform=inference_transform,
        grid_size=args.grid_size,
        lat_mean=lat_mean,
        lat_std=lat_std,
        lon_mean=lon_mean,
        lon_std=lon_std,
        lat_edges=lat_edges,
        lon_edges=lon_edges,
    )

    test_dataset = GPSDataset(
        test_df,
        transform=inference_transform,
        grid_size=args.grid_size,
        lat_mean=lat_mean,
        lat_std=lat_std,
        lon_mean=lon_mean,
        lon_std=lon_std,
        lat_edges=lat_edges,
        lon_edges=lon_edges,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = Model(num_cells=num_cells)
    initialize_pretrained_backbone(model)
    set_model_normalization(model, lat_mean, lat_std, lon_mean, lon_std)
    model = model.to(device)

    coord_criterion = nn.SmoothL1Loss()
    cell_criterion = nn.CrossEntropyLoss()

    optimizer = make_optimizer(
        model,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_val_rmse = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0

    print("\n========== TRAINING ==========")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            coord_criterion,
            cell_criterion,
            device,
            cell_loss_weight=args.cell_loss_weight,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            lat_mean,
            lat_std,
            lon_mean,
            lon_std,
        )

        val_rmse = float(val_metrics["rmse_m"])
        scheduler.step(val_rmse)

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, "
            f"val_RMSE={val_rmse:.2f} m, "
            f"val_avg={val_metrics['avg_m']:.2f} m, "
            f"val_median={val_metrics['median_m']:.2f} m"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print("Early stopping.")
            break

    print("\nBest validation RMSE:", best_val_rmse)
    print("Best epoch:", best_epoch)

    if best_state is None:
        raise RuntimeError("No best model state was saved. Something went wrong during training.")

    model.load_state_dict(best_state)

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        lat_mean,
        lat_std,
        lon_mean,
        lon_std,
    )

    print("\nFINAL TEST RESULTS")
    print(f"Test RMSE: {test_metrics['rmse_m']:.2f} m")
    print(f"Test Avg Distance: {test_metrics['avg_m']:.2f} m")
    print(f"Test Median Distance: {test_metrics['median_m']:.2f} m")

    torch.save(model.state_dict(), args.out)
    print(f"\nSaved final model to: {args.out}")

    preds = test_metrics["preds"]
    actuals = test_metrics["actuals"]
    distances = test_metrics["distances"]

    pred_df = pd.DataFrame(
        {
            "actual_lat": actuals[:, 0],
            "actual_lon": actuals[:, 1],
            "pred_lat": preds[:, 0],
            "pred_lon": preds[:, 1],
            "distance_m": distances,
        }
    )

    pred_df.to_csv("final_test_predictions.csv", index=False)
    print("Saved final_test_predictions.csv")

    summary = {
        "best_epoch": best_epoch,
        "best_val_rmse_m": best_val_rmse,
        "test_rmse_m": float(test_metrics["rmse_m"]),
        "test_avg_m": float(test_metrics["avg_m"]),
        "test_median_m": float(test_metrics["median_m"]),
        "image_size": args.image_size,
        "grid_size": args.grid_size,
        "num_cells": num_cells,
        "cell_loss_weight": args.cell_loss_weight,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "val_size": args.val_size,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
    }

    pd.DataFrame([summary]).to_csv("training_summary.csv", index=False)
    print("Saved training_summary.csv")


if __name__ == "__main__":
    main()