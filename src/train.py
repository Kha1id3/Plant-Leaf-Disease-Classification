#!/usr/bin/env python3
import argparse
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from datasets import ImageCsvDataset, classes_from_any_csv
from models import build_model
from utils import set_all_seeds, compute_class_weights

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(train: bool):
    if train:
        return T.Compose([
            T.Resize(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def epoch_run(
    loader,
    model,
    criterion,
    optimizer=None,
    device="cpu",
    amp_enabled=False,
    scaler=None,
    channels_last=False
):
    """One full pass over a loader. Supports AMP and channels_last."""
    is_train = optimizer is not None
    model.train(mode=is_train)

    losses = []
    all_y, all_pred = [], []

    for x, y, _ in loader:
        
        if channels_last and x.ndim == 4:
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        amp_ctx = torch.amp.autocast("cuda", enabled=amp_enabled) if device.type == "cuda" else nullcontext()
        with torch.set_grad_enabled(is_train):
            with amp_ctx:
                logits = model(x)
                loss = criterion(logits, y)

        if is_train:
            if amp_enabled and scaler is not None:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        pred = torch.argmax(logits, dim=1)
        all_y.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    all_y = np.concatenate(all_y) if all_y else np.array([])
    all_pred = np.concatenate(all_pred) if all_pred else np.array([])

    if all_y.size == 0:
        return float(np.mean(losses) if losses else 0.0), 0.0, 0.0, 0.0, 0.0, all_y, all_pred

    acc = accuracy_score(all_y, all_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_y, all_pred, average="macro", zero_division=0
    )
    return float(np.mean(losses)), acc, prec, rec, f1, all_y, all_pred


def train_one(config):
    set_all_seeds(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    if device.type == "cuda" and config.fast_cudnn:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

   
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and config.amp)) if device.type == "cuda" else None

  
    classes, class_to_idx = classes_from_any_csv([config.train_csv, config.val_csv])

 
    ds_train = ImageCsvDataset(config.train_csv, class_to_idx, transform=build_transforms(train=True))
    ds_val   = ImageCsvDataset(config.val_csv,   class_to_idx, transform=build_transforms(train=False))

    
    pw = config.workers > 0
    train_loader = DataLoader(
        ds_train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=pw,
        prefetch_factor=(config.prefetch if pw else None),
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=pw,
        prefetch_factor=(config.prefetch if pw else None),
    )


    model, tag = build_model(config.backbone, num_classes=len(classes), pretrained=config.pretrained)
    model = model.to(device)
    if device.type == "cuda" and config.channels_last:
        model = model.to(memory_format=torch.channels_last)

   
    if config.class_weight:
        weights = compute_class_weights(config.train_csv, class_to_idx).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

   
    lr = config.lr_tl if config.pretrained else config.lr_scratch
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    
    head_only_epochs = config.warmup_head_epochs if config.pretrained else 0
    if head_only_epochs > 0:
        for p in model.parameters():
            p.requires_grad = False
        if hasattr(model, "fc"):      
            for p in model.fc.parameters():
                p.requires_grad = True
        else:                       
            for p in model.classifier.parameters():
                p.requires_grad = True

    best_f1 = -1.0
    best_state = None
    history = []
    max_epochs = config.epochs_tl if config.pretrained else config.epochs_scratch
    patience = config.patience
    since_best = 0

    for epoch in range(1, max_epochs + 1):
      
        if config.pretrained and epoch == head_only_epochs + 1:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        tr = epoch_run(
            train_loader, model, criterion, optimizer, device,
            amp_enabled=(device.type == "cuda" and config.amp),
            scaler=scaler, channels_last=config.channels_last
        )
        vl = epoch_run(
            val_loader, model, criterion, None, device,
            amp_enabled=False, scaler=None, channels_last=config.channels_last
        )
        train_loss, train_acc, train_prec, train_rec, train_f1, _, _ = tr
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = vl

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "train_prec": train_prec, "train_rec": train_rec, "train_f1": train_f1,
            "val_loss": val_loss, "val_acc": val_acc,
            "val_prec": val_prec, "val_rec": val_rec, "val_f1": val_f1,
        })
        print(f"Epoch {epoch:03d} | train f1={train_f1:.4f}  val f1={val_f1:.4f}")

        improved = val_f1 > best_f1 + 1e-6
        if improved:
            best_f1 = val_f1
            best_state = model.state_dict()
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                print("Early stopping.")
                break

   
    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{tag}_seed{config.seed}"

    ckpt_path = outdir / f"{run_tag}.pt"
    torch.save({
        "model_state": best_state,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "backbone": config.backbone,
        "pretrained": config.pretrained,
    }, ckpt_path)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(outdir / f"{run_tag}_history.csv", index=False)

    print("Saved:", ckpt_path)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["resnet18", "efficientnet_b0"], required=True)
    ap.add_argument("--pretrained", action="store_true", help="Use ImageNet weights (transfer learning)")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--outdir", default="artifacts/models")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)  
    ap.add_argument("--epochs_tl", type=int, default=30)
    ap.add_argument("--epochs_scratch", type=int, default=40)
    ap.add_argument("--warmup_head_epochs", type=int, default=4)
    ap.add_argument("--lr_tl", type=float, default=3e-4)
    ap.add_argument("--lr_scratch", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--class_weight", action="store_true")

   
    ap.add_argument("--amp", action="store_true", help="Mixed precision (CUDA only)")
    ap.add_argument("--channels_last", action="store_true", help="Use channels_last memory format (CUDA only)")
    ap.add_argument("--prefetch", type=int, default=2, help="DataLoader prefetch_factor (requires workers>0)")
    ap.add_argument("--fast_cudnn", action="store_true", help="Enable cudnn.benchmark (non-deterministic, faster)")

    return ap.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    train_one(cfg)
