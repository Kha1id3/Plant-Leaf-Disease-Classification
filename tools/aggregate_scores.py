#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import pandas as pd

TAGS = ["resnet18_tl","resnet18_scratch","effnet_b0_tl","effnet_b0_scratch"]

def read_summary(p):
    # lines: acc=..., macroF1=..., precision=..., recall=...
    d = {}
    with open(p, "r") as f:
        for line in f:
            k,v = line.strip().split("=")
            d[k] = float(v)
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/eval")
    ap.add_argument("--seeds", nargs="*", type=int, default=[13,17,23])
    ap.add_argument("--out_csv", default="artifacts/scores_by_seed.csv")
    args = ap.parse_args()

    rows = []
    for s in args.seeds:
        for tag in TAGS:
            summ = Path(args.root)/f"seed{s}"/f"{tag}_summary.txt"
            if summ.exists():
                d = read_summary(summ)
                rows.append({
                    "seed": s,
                    "tag": tag,
                    "macroF1": d["macroF1"],
                    "accuracy": d["acc"],
                    "precision": d["precision"],
                    "recall": d["recall"],
                })
            else:
                print("MISSING:", summ)

    long_df = pd.DataFrame(rows).sort_values(["seed","tag"])
    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out, index=False)

    # also write a wide format with F1 columns for stats_tests
    wide = long_df.pivot(index="seed", columns="tag", values="macroF1")
    wide.columns = [f"f1_{c}" for c in wide.columns]
    wide.reset_index().to_csv(out.parent/"scores_by_seed_wide.csv", index=False)

    print("Saved:", out, "and", out.parent/"scores_by_seed_wide.csv")

if __name__ == "__main__":
    main()
