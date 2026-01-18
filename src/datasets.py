from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageCsvDataset(Dataset):
    """
    Dataset that reads image filepaths and class labels from a CSV with columns: filepath,class
    """
    def __init__(self, csv_path: str, class_to_idx: Dict[str, int], transform=None):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        needed = {"filepath", "class"}
        if not needed.issubset(self.df.columns):
            raise ValueError(f"{csv_path} must have columns: {needed}")
        self.samples = [(row.filepath, class_to_idx[row["class"]]) for _, row in self.df.iterrows()]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, path


def read_fixed_test_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"filepath", "class", "split"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{csv_path} must have columns: {needed}")
    return df


def classes_from_any_csv(csvs: List[str]) -> Tuple[List[str], Dict[str, int]]:
    labels = set()
    for p in csvs:
        df = pd.read_csv(p)
        col = "class" if "class" in df.columns else None
        if col is None:
            raise ValueError(f"{p} must contain a 'class' column")
        labels.update(df["class"].unique().tolist())
    classes = sorted(list(labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return classes, class_to_idx
