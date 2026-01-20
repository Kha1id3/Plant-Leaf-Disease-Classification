#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed_csv", required=True, help="Path to fixed_test_split.csv")
    ap.add_argument("--outdir", required=True, help="Output dir for generated CSVs")
    ap.add_argument("--seeds", nargs="*", type=int, default=[13, 17, 23])
    ap.add_argument("--val_size", type=float, default=0.2, help="Validation fraction of trainval")
    args = ap.parse_args()

    fixed = pd.read_csv(args.fixed_csv)
    if not {"filepath", "class", "split"}.issubset(fixed.columns):
        raise ValueError("fixed_csv must have columns: filepath,class,split")

    tv = fixed[fixed["split"] == "trainval"].copy()
    X = tv["filepath"].values
    y = tv["class"].values

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=seed)
        (train_idx, val_idx), = sss.split(X, y)
        train = tv.iloc[train_idx][["filepath", "class"]].copy()
        val = tv.iloc[val_idx][["filepath", "class"]].copy()
        train.to_csv(outdir / f"train_seed{seed}.csv", index=False)
        val.to_csv(outdir / f"val_seed{seed}.csv", index=False)
        print(f"Wrote: {outdir / f'train_seed{seed}.csv'} and {outdir / f'val_seed{seed}.csv'}")

if __name__ == "__main__":
    main()
