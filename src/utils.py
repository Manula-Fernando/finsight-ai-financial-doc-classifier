from __future__ import annotations
import json
import os
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from . import config


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}


def save_confusion_matrix(y_true: List[int], y_pred: List[int], class_names: List[str], out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def save_label_map(classes: List[str], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({i: c for i, c in enumerate(classes)}, f, indent=2)


def load_label_map(path: str) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # ensure keys are ints
    return {int(k): v for k, v in data.items()}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
