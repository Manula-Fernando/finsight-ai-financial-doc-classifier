from __future__ import annotations
import argparse
import os
from typing import Tuple, List

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math

from . import config
from .model import DocClassifier
from .utils import load_label_map, get_device


def build_infer_transform(img_size: int) -> A.Compose:
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        # use 'fill' per Albumentations v2 API
        A.PadIfNeeded(img_size, img_size, border_mode=0, fill=255),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def _pil_to_tensor(pil_img: Image.Image, img_size: int, device: torch.device) -> torch.Tensor:
    tfm = build_infer_transform(img_size)
    arr = np.array(pil_img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    t = tfm(image=arr)["image"].unsqueeze(0).to(device)
    return t


def predict_image(model: DocClassifier, image_path: str, device: torch.device, img_size: int, classes: List[str], tta: int = 0) -> Tuple[str, float]:
    """Predict a single image returning (label, confidence).

    tta: number of test-time augmentation samples (0 = disabled). Uses deterministic set of
    rotations + slight brightness shifts for doc robustness (no retrain needed).
    """
    base = Image.open(image_path).convert("RGB")
    variants: List[Image.Image] = [base]
    if tta > 0:
        # deterministic variants: rotations and light brightness adjustments
        try:
            from PIL import ImageEnhance
            rots = [0, 90, 180, 270]
            bris = [1.0, 0.9, 1.1]
            for r in rots:
                rb = base.rotate(r, expand=True) if r != 0 else base
                for b in bris:
                    if r == 0 and math.isclose(b, 1.0):
                        continue
                    variants.append(ImageEnhance.Brightness(rb).enhance(b))
        except Exception:
            pass
        # limit to requested count (include original first)
        if len(variants) > (tta + 1):
            variants = variants[: tta + 1]

    model.eval()
    with torch.no_grad():
        probs_accum = torch.zeros((1, len(classes)), device=device)
        for v in variants:
            x = _pil_to_tensor(v, img_size, device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            probs_accum += probs
        probs_mean = probs_accum / max(1, len(variants))
        probs_vec = probs_mean[0]
        conf, idx = torch.max(probs_vec, dim=0)
    return classes[int(idx.item())], float(conf.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--weights", type=str, default=os.path.join(config.MODELS_DIR, "best.pt"))
    parser.add_argument("--img_size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--tta", type=int, default=0, help="Number of TTA variants (rotations/brightness combos) to sample (0=off).")
    args = parser.parse_args()

    device = get_device()

    label_map_path = os.path.join(config.MODELS_DIR, "label_map.json")
    id2label = load_label_map(label_map_path)

    ckpt = torch.load(args.weights, map_location=device)
    classes = ckpt.get("classes", [id2label[i] for i in sorted(id2label)])

    model = DocClassifier(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    label, conf = predict_image(model, args.image, device, args.img_size, classes, tta=args.tta)
    print({"label": label, "confidence": round(conf, 4)})


if __name__ == "__main__":
    main()
