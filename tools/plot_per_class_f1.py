#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_class_csv", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--title", default="Per-class F1")
    args = ap.parse_args()

    df = pd.read_csv(args.per_class_csv)
    df = df.sort_values("f1")
    y = df["class_name"].values
    x = df["f1"].values

    plt.figure(figsize=(8, max(6, 0.35*len(y))))
    plt.barh(y, x)
    plt.xlabel("F1")
    plt.title(args.title)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print("Saved:", args.out_png)

if __name__ == "__main__":
    main()
