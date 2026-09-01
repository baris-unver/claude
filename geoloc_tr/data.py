"""Datasets and transforms."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .aerial import AerialPatchStore
from .config import Config
from .geo import CellHierarchy, cell_ids, cells_centers

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(size: int, strength: str = "strong") -> T.Compose:
    if strength == "strong":  # real street-level imagery
        ops = [T.RandomResizedCrop(size, scale=(0.4, 1.0), ratio=(0.75, 1.33)), T.ColorJitter(0.3, 0.3, 0.2, 0.05),
               T.RandomGrayscale(0.05)]
    elif strength == "light":  # synthetic / smoke tests
        ops = [T.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.9, 1.1)), T.ColorJitter(0.1, 0.1, 0.0, 0.0)]
    else:
        raise ValueError(f"unknown augmentation strength {strength!r}")
    return T.Compose(ops + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def eval_transform(size: int) -> T.Compose:
    return T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def aerial_transform(size: int, train: bool) -> T.Compose:
    ops = [T.Resize(size)]
    if train:
        ops += [T.RandomResizedCrop(size, scale=(0.7, 1.0)), T.ColorJitter(0.2, 0.2, 0.1)]
    else:
        ops += [T.CenterCrop(size)]
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


def load_table(cfg: Config, splits: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(cfg.splits_path)
    df = df[df["keep"]]
    if splits:
        df = df[df["split"].isin(splits)]
    return df.reset_index(drop=True)


class GroundDataset(Dataset):
    """Street-level images with hierarchical S2 labels (and, optionally, the aerial patch of their cell)."""

    def __init__(self, df: pd.DataFrame, hierarchy: CellHierarchy, image_size: int, train: bool,
                 aerial: AerialPatchStore | None = None, db_level: int | None = None, augment: str = "strong"):
        self.df = df.reset_index(drop=True)
        self.paths = self.df["path"].tolist()
        self.lat = self.df["lat"].to_numpy(np.float64)
        self.lon = self.df["lon"].to_numpy(np.float64)
        self.labels = hierarchy.labels(self.lat, self.lon)
        self.tf = train_transform(image_size, augment) if train else eval_transform(image_size)
        self.aerial = aerial
        self.aerial_tf = aerial_transform(image_size, train)
        if aerial is not None:
            assert db_level is not None
            self.db_cells = cell_ids(self.lat, self.lon, db_level)
            self.db_centers = cells_centers(self.db_cells)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        img = Image.open(self.paths[i]).convert("RGB")
        out = {
            "image": self.tf(img),
            "labels": torch.from_numpy(self.labels[i]),
            "latlon": torch.tensor([self.lat[i], self.lon[i]], dtype=torch.float64),
            "index": i,
        }
        if self.aerial is not None:
            c = int(self.db_cells[i])
            patch = self.aerial.get(c, float(self.db_centers[i, 0]), float(self.db_centers[i, 1]))
            out["aerial"] = self.aerial_tf(patch)
            out["cell"] = torch.tensor(c, dtype=torch.int64)
        return out


class ImageListDataset(Dataset):
    def __init__(self, paths: list[str | Path], image_size: int):
        self.paths = [str(p) for p in paths]
        self.tf = eval_transform(image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return {"image": self.tf(Image.open(self.paths[i]).convert("RGB")), "index": i}


class AerialCellDataset(Dataset):
    def __init__(self, cells: np.ndarray, centers: np.ndarray, store: AerialPatchStore, image_size: int):
        self.cells, self.centers, self.store = cells, centers, store
        self.tf = aerial_transform(image_size, train=False)

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, i):
        patch = self.store.get(int(self.cells[i]), float(self.centers[i, 0]), float(self.centers[i, 1]))
        return {"aerial": self.tf(patch), "index": i}
