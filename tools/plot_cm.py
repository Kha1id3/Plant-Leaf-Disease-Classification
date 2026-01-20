#!/usr/bin/env python3
import argparse, numpy as np, torch, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cm_npy", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_png", required=True)
    args = ap.parse_args()

    cm = np.load(args.cm_npy)
    classes = torch.load(args.ckpt, map_location="cpu").get("classes", [str(i) for i in range(cm.shape[0])])

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix")
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=90)
    plt.yticks(ticks, classes)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print("Saved:", args.out_png)

if __name__ == "__main__":
    main()
