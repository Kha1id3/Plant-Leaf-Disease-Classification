#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)

from datasets import ImageCsvDataset, read_fixed_test_csv, classes_from_any_csv
from models import build_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_transforms():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to model .pt file")
    ap.add_argument("--fixed_csv", required=True, help="Path to fixed_test_split.csv")
    ap.add_argument("--outdir", default="artifacts/eval")
    ap.add_argument("--backbone", choices=["resnet18", "efficientnet_b0"], required=True)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)  # Windows-safe
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fixed = read_fixed_test_csv(args.fixed_csv)
    df_test = fixed[fixed["split"] == "test"][["filepath","class"]].copy()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    tmp_csv = outdir / "_tmp_test.csv"
    df_test.to_csv(tmp_csv, index=False)

    # classes order: try checkpoint first for exact mapping, else derive from csv
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "classes" in ckpt and ckpt["classes"] is not None:
        classes = ckpt["classes"]
        class_to_idx = ckpt.get("class_to_idx", {c: i for i, c in enumerate(classes)})
    else:
        classes, class_to_idx = classes_from_any_csv([str(tmp_csv)])

    ds_test = ImageCsvDataset(str(tmp_csv), class_to_idx, transform=build_transforms())
    loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=(device.type=="cuda"))

    model, tag = build_model(args.backbone, num_classes=len(classes), pretrained=args.pretrained)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model = model.to(device).eval()

    all_y, all_pred, all_paths = [], [], []
    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            all_pred.append(pred)
            all_y.append(y.numpy())
            all_paths.extend(paths)

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    np.save(outdir / f"{tag}_cm.npy", cm)
    pd.DataFrame({
        "filepath": all_paths,
        "y_true": y_true,
        "y_pred": y_pred,
    }).to_csv(outdir / f"{tag}_test_preds.csv", index=False)

    with open(outdir / f"{tag}_summary.txt", "w") as f:
        f.write(f"acc={acc:.4f}\nmacroF1={f1:.4f}\nprecision={prec:.4f}\nrecall={rec:.4f}\n")

    print(f"Test macro-F1={f1:.4f}  acc={acc:.4f}. Saved artifacts to {outdir}.")

if __name__ == "__main__":
    main()
