#!/usr/bin/env python3
import argparse, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history_csv", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--title", default="Training/Validation Curves")
    args = ap.parse_args()

    df = pd.read_csv(args.history_csv)

    # Plot F1
    plt.figure(figsize=(7,4))
    plt.plot(df["epoch"], df["train_f1"], label="train F1")
    plt.plot(df["epoch"], df["val_f1"],   label="val F1")
    plt.xlabel("Epoch"); plt.ylabel("F1"); plt.title(args.title + " (F1)")
    plt.legend(); plt.tight_layout()
    f1_png = args.out_png.replace(".png", "_f1.png")
    plt.savefig(f1_png, dpi=200)
    print("Saved:", f1_png)

    # Plot loss
    plt.figure(figsize=(7,4))
    plt.plot(df["epoch"], df["train_loss"], label="train loss")
    plt.plot(df["epoch"], df["val_loss"],   label="val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(args.title + " (Loss)")
    plt.legend(); plt.tight_layout()
    loss_png = args.out_png.replace(".png", "_loss.png")
    plt.savefig(loss_png, dpi=200)
    print("Saved:", loss_png)

if __name__ == "__main__":
    main()
