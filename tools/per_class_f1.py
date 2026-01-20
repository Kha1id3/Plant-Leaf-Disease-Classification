#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd, torch
from sklearn.metrics import precision_recall_fscore_support

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_csv", required=True)
    ap.add_argument("--ckpt", required=True, help="same ckpt used for eval (to get class names)")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.preds_csv)
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt.get("classes", [str(i) for i in range(int(max(y_true.max(), y_pred.max()))+1)])

    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(classes)), zero_division=0)
    out = pd.DataFrame({
        "class_idx": list(range(len(classes))),
        "class_name": classes,
        "precision": prec, "recall": rec, "f1": f1, "support": support
    })
    out.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)

if __name__ == "__main__":
    main()
