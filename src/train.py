from __future__ import annotations
import argparse
import os
from typing import List
from datetime import datetime

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from . import config
from .datasets import get_loaders, discover_classes
from .model import DocClassifier
from .utils import compute_metrics, save_confusion_matrix, save_label_map, set_seed, get_device
from sklearn.metrics import classification_report
import json
import subprocess
import hashlib


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for xb, yb, _ in tqdm(loader, desc="train", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(yb.detach().cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_targets, all_preds)
    return avg_loss, metrics


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb, _ in tqdm(loader, desc="eval", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_targets.extend(yb.detach().cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_targets, all_preds)
    return avg_loss, metrics, all_targets, all_preds


def freeze_backbone(model: DocClassifier, freeze: bool = True):
    requires = not freeze
    for p in model.backbone.parameters():
        p.requires_grad = requires


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--img_size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--classes", nargs="+", default=config.CLASS_NAMES)
    parser.add_argument("--freeze_epochs", type=int, default=2)
    parser.add_argument("--run_name", type=str, default=None, help="Optional MLflow run name. If omitted, a professional name is auto-generated.")
    args = parser.parse_args()

    set_seed(config.RANDOM_STATE)
    device = get_device()

    # Determine classes: honor explicit list unless it's 'auto' or missing in data
    requested: List[str] = [c.lower() for c in args.classes]
    data_classes = discover_classes(config.RAW_DIR, min_images=1)
    if requested == ["auto"] or not set(requested).issubset(set(data_classes)):
        classes: List[str] = sorted(data_classes)
    else:
        classes = requested

    train_loader, val_loader, test_loader = get_loaders(args.img_size, args.batch_size, config.NUM_WORKERS, classes)

    model = DocClassifier(num_classes=len(classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # freeze backbone initially
    freeze_backbone(model, True)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    label_map_path = os.path.join(config.MODELS_DIR, "label_map.json")
    save_label_map(classes, label_map_path)

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    best_val = 0.0
    best_path = os.path.join(config.MODELS_DIR, "best.pt")

    # Build professional default run name if none supplied
    if args.run_name:
        run_name = args.run_name
    else:
        ts = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
        run_name = (
            f"FinSightClassifier-{len(classes)}cls-"
            f"{config.BACKBONE.replace('tf_efficientnet_', 'effnet').replace('_ns','')}-"
            f"{args.img_size}px-{ts}"
        )

    with mlflow.start_run(run_name=run_name):
        # Capture git commit hash (if repo) for reproducibility
        git_hash = None
        try:
            git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.ROOT_DIR).decode().strip()
        except Exception:
            pass
        mlflow.log_params({
            "backbone": config.BACKBONE,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "classes": ",".join(classes),
            "run_name": run_name,
            "freeze_epochs": args.freeze_epochs,
            "weight_decay": args.weight_decay,
        })
        if git_hash:
            mlflow.set_tags({"git_commit": git_hash})
        mlflow.set_tags({
            "project": "FinSight Document Classifier",
            "owner": "Manula Fernando",
            "privacy_mode": "in-memory-ocr-available",
        })
        # log label map as artifact
        if os.path.exists(label_map_path):
            mlflow.log_artifact(label_map_path, artifact_path="metadata")
        # For curves
        hist = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_f1": [], "val_f1": []}
        for epoch in range(1, args.epochs + 1):
            # Unfreeze after freeze_epochs
            if epoch == args.freeze_epochs + 1:
                freeze_backbone(model, False)
                optimizer = AdamW(model.parameters(), lr=args.lr * 0.3, weight_decay=args.weight_decay)
            tr_loss, tr_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_metrics, y_true, y_pred = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": tr_loss,
                "train_acc": tr_metrics["accuracy"],
                "train_f1": tr_metrics["f1_macro"],
                "val_loss": val_loss,
                "val_acc": val_metrics["accuracy"],
                "val_f1": val_metrics["f1_macro"],
            }, step=epoch)
            hist["epoch"].append(epoch)
            hist["train_loss"].append(tr_loss)
            hist["val_loss"].append(val_loss)
            hist["train_acc"].append(tr_metrics["accuracy"])
            hist["val_acc"].append(val_metrics["accuracy"])
            hist["train_f1"].append(tr_metrics["f1_macro"])
            hist["val_f1"].append(val_metrics["f1_macro"])

            # save best
            if val_metrics["f1_macro"] > best_val:
                best_val = val_metrics["f1_macro"]
                torch.save({"model_state": model.state_dict(), "classes": classes}, best_path)
                mlflow.log_artifact(best_path, artifact_path="models")
                # confusion matrix plot
                cm_path = os.path.join(config.OUTPUTS_DIR, "confusion_matrix_val.png")
                save_confusion_matrix(y_true, y_pred, classes, cm_path)
                mlflow.log_artifact(cm_path, artifact_path="plots")

        # evaluate on test set with best weights
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
        test_loss, test_metrics, y_true_t, y_pred_t = evaluate(model, test_loader, criterion, device)
        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_acc": test_metrics["accuracy"],
            "test_f1": test_metrics["f1_macro"],
        })
        cm_test_path = os.path.join(config.OUTPUTS_DIR, "confusion_matrix_test.png")
        save_confusion_matrix(y_true_t, y_pred_t, classes, cm_test_path)
        mlflow.log_artifact(cm_test_path, artifact_path="plots")

        # save detailed metrics & predictions
        report_val = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
        report_test = classification_report(y_true_t, y_pred_t, target_names=classes, output_dict=True)
        metrics_out = {
            "params": {"backbone": config.BACKBONE, "img_size": args.img_size, "epochs": args.epochs, "batch_size": args.batch_size},
            "val": {"loss": val_loss, **val_metrics, "report": report_val},
            "test": {"loss": test_loss, **test_metrics, "report": report_test}
        }
        os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
        with open(os.path.join(config.OUTPUTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_out, f, indent=2)
        import pandas as pd
        preds_df = pd.DataFrame({"y_true": y_true_t, "y_pred": y_pred_t})
        preds_df.to_csv(os.path.join(config.OUTPUTS_DIR, "test_predictions.csv"), index=False)
        # Run metadata summary (for app display)
        run_info = {
            "run_name": run_name,
            "git_commit": git_hash,
            "classes": classes,
            "n_classes": len(classes),
            "backbone": config.BACKBONE,
            "img_size": args.img_size,
            "best_val_f1": best_val,
            "test_f1": test_metrics.get("f1_macro"),
        }
        with open(os.path.join(config.OUTPUTS_DIR, "run_info.json"), "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2)
        mlflow.log_artifact(os.path.join(config.OUTPUTS_DIR, "run_info.json"), artifact_path="metadata")

        # Plot and log training curves
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            ax[0].plot(hist["epoch"], hist["train_loss"], label="train")
            ax[0].plot(hist["epoch"], hist["val_loss"], label="val")
            ax[0].set_title("Loss")
            ax[0].set_xlabel("Epoch")
            ax[0].legend()
            ax[1].plot(hist["epoch"], hist["train_acc"], label="train")
            ax[1].plot(hist["epoch"], hist["val_acc"], label="val")
            ax[1].set_title("Accuracy")
            ax[1].set_xlabel("Epoch")
            ax[1].legend()
            ax[2].plot(hist["epoch"], hist["train_f1"], label="train")
            ax[2].plot(hist["epoch"], hist["val_f1"], label="val")
            ax[2].set_title("F1 (macro)")
            ax[2].set_xlabel("Epoch")
            ax[2].legend()
            curves_path = os.path.join(config.OUTPUTS_DIR, "training_curves.png")
            fig.tight_layout()
            fig.savefig(curves_path)
            plt.close(fig)
            mlflow.log_artifact(curves_path, artifact_path="plots")
        except Exception as e:
            print("[warn] Failed to plot training curves:", e)

        # log model
        mlflow.pytorch.log_model(model, artifact_path="pytorch-model")

    print("Training complete. Best F1 (val):", best_val)


if __name__ == "__main__":
    main()
