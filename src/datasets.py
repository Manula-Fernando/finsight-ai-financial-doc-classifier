from __future__ import annotations
import os
from typing import Callable, List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from . import config


def build_transforms(img_size: int, is_train: bool) -> A.Compose:
    if is_train:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            # PadIfNeeded now uses 'fill' instead of deprecated 'value'
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, fill=255),
            # ShiftScaleRotate deprecated in v2 -> replace with Affine
            A.Affine(
                scale=(0.95, 1.05),  # mirrors scale_limit=0.05
                translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},  # shift_limit=0.01
                rotate=(-3, 3),
                border_mode=cv2.BORDER_CONSTANT,
                fill=255,
                p=0.5,
            ),
            A.RandomBrightnessContrast(0.05, 0.05, p=0.3),
            # GaussNoise now expects std_range in [0,1]; approximate previous intensity (5-15 variance ~ std 2-4) by small fractional noise
            A.GaussNoise(std_range=(0.01, 0.03), mean_range=(0.0, 0.0), p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, fill=255),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class DocImageDataset(Dataset):
    def __init__(self, root: str, classes: List[str], split: str, transform: A.Compose):
        self.samples: List[Tuple[str, int]] = []
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for cls in classes:
            class_dir = os.path.join(root, cls)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), self.class_to_idx[cls]))

        # simple split by index
        rng = np.random.RandomState(config.RANDOM_STATE)
        rng.shuffle(self.samples)
        n = len(self.samples)
        n_train = int(0.7 * n)
        n_val = int(0.85 * n)
        if split == "train":
            self.samples = self.samples[:n_train]
        elif split == "val":
            self.samples = self.samples[n_train:n_val]
        else:
            self.samples = self.samples[n_val:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)
        # if grayscale, still 3ch
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        aug = self.transform(image=img_np)
        x = aug["image"]  # C,H,W tensor
        y = torch.tensor(label, dtype=torch.long)
        return x, y, path


def get_loaders(img_size: int, batch_size: int, num_workers: int, classes: List[str]):
    tf_train = build_transforms(img_size, True)
    tf_val = build_transforms(img_size, False)
    train_ds = DocImageDataset(config.RAW_DIR, classes, "train", tf_train)
    val_ds = DocImageDataset(config.RAW_DIR, classes, "val", tf_val)
    test_ds = DocImageDataset(config.RAW_DIR, classes, "test", tf_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def discover_classes(root: str, min_images: int = 1) -> List[str]:
    classes: List[str] = []
    if not os.path.isdir(root):
        return classes
    for d in sorted(os.listdir(root)):
        full = os.path.join(root, d)
        if not os.path.isdir(full):
            continue
        n = len([f for f in os.listdir(full) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        if n >= min_images:
            classes.append(d)
    return classes
