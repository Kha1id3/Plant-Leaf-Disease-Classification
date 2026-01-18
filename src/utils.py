import random, os
from typing import Dict
import numpy as np
import torch
import pandas as pd
from collections import Counter

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_class_weights(train_csv: str, class_to_idx: Dict[str, int]) -> torch.Tensor:
    """
    Inverse-frequency class weights, normalized to sum to number of classes.
    """
    df = pd.read_csv(train_csv)
    counts = Counter(df["class"].values.tolist())
    weights = np.zeros(len(class_to_idx), dtype=np.float32)
    for c, idx in class_to_idx.items():
        weights[idx] = 1.0 / max(1, counts.get(c, 0))
    # Normalize for stability
    weights = weights * (len(class_to_idx) / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)
