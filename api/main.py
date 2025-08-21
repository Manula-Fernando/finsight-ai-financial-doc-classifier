from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
import io
import os
import torch
import uvicorn

from src import config
from src.model import DocClassifier
from src.infer import build_infer_transform
from src.utils import load_label_map, get_device
from src import ocr as ocr_mod

app = FastAPI(title="FinSight AI API", version="0.1.0")

_model = None
_classes: List[str] = []
_device = None


def load_model():
    global _model, _classes, _device
    if _model is not None:
        return _model, _classes, _device
    _device = get_device()
    label_map_path = os.path.join(config.MODELS_DIR, "label_map.json")
    id2label = load_label_map(label_map_path)
    _classes = [id2label[i] for i in sorted(id2label)]
    weights = os.path.join(config.MODELS_DIR, "best.pt")
    ckpt = torch.load(weights, map_location=_device)
    model = DocClassifier(num_classes=len(_classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(_device)
    model.eval()
    _model = model
    return _model, _classes, _device


@app.on_event("startup")
async def startup_event():
    load_model()


@app.post("/classify")
async def classify(files: List[UploadFile] = File(...), include_full_text: bool = True):
    model, classes, device = load_model()
    tfm = build_infer_transform(config.IMG_SIZE)
    results = []
    for f in files:
        content = await f.read()
        pil = Image.open(io.BytesIO(content)).convert("RGB")
        import numpy as np
        arr = np.array(pil)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        x = tfm(image=arr)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        top_idx = int(probs.argmax())
        pred_label = classes[top_idx]

        # OCR (in-memory privacy path)
        import importlib, inspect
        ocr_live = importlib.reload(ocr_mod)
        fn = ocr_live.extract_fields_from_pil
        params = inspect.signature(fn).parameters
        if 'include_full_text' in params:
            fields = fn(pil, include_full_text=include_full_text)
        else:
            fields = fn(pil)
        results.append({
            "filename": f.filename,
            "prediction": pred_label,
            "confidence": float(probs[top_idx]),
            "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))},
            "ocr": fields
        })
    return JSONResponse(results)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000)
