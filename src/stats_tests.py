#!/usr/bin/env python3
"""
Statistical tests for the report:
- Wilcoxon signed-rank test on macro-F1 across seeds.
- McNemar's test from paired predictions on the fixed test set.
"""
import argparse
import pandas as pd
from scipy.stats import wilcoxon, binomtest

def wilcoxon_from_table(csv_path: str, col_a: str, col_b: str):
    df = pd.read_csv(csv_path)
    if col_a not in df.columns or col_b not in df.columns:
        raise ValueError(f"CSV must contain {col_a} and {col_b} columns.")
    stat, p = wilcoxon(df[col_a], df[col_b], zero_method="wilcox", alternative="two-sided")
    print(f"Wilcoxon({col_a} vs {col_b}): stat={stat:.3f}, p={p:.6f}")

def mcnemar_from_preds(csv_a: str, csv_b: str):
    a = pd.read_csv(csv_a)
    b = pd.read_csv(csv_b)
    need = {"filepath","y_true","y_pred"}
    if not need.issubset(a.columns) or not {"filepath","y_pred"}.issubset(b.columns):
        raise ValueError("Prediction CSVs must have columns: 'filepath','y_true','y_pred' (y_true only needed in A).")
    merged = a[["filepath","y_true","y_pred"]].merge(b[["filepath","y_pred"]], on="filepath", suffixes=("_a","_b"))
    b_ct = ((merged["y_pred_a"] == merged["y_true"]) & (merged["y_pred_b"] != merged["y_true"])).sum()
    c_ct = ((merged["y_pred_a"] != merged["y_true"]) & (merged["y_pred_b"] == merged["y_true"])).sum()
    n = b_ct + c_ct
    if n == 0:
        print("McNemar: models have identical errors (n=0)")
        return
    p = binomtest(k=min(b_ct, c_ct), n=n, p=0.5, alternative="two-sided").pvalue
    print(f"McNemar exact: b={b_ct}, c={c_ct}, n={n}, p={p:.6f}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("wilcoxon")
    w.add_argument("--csv", required=True, help="CSV with per-seed scores")
    w.add_argument("--col_a", required=True)
    w.add_argument("--col_b", required=True)

    m = sub.add_parser("mcnemar")
    m.add_argument("--preds_a", required=True)
    m.add_argument("--preds_b", required=True)

    args = ap.parse_args()
    if args.cmd == "wilcoxon":
        wilcoxon_from_table(args.csv, args.col_a, args.col_b)
    else:
        mcnemar_from_preds(args.preds_a, args.preds_b)

if __name__ == "__main__":
    main()
