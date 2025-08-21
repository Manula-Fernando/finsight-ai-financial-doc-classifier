import os
import io
import json
import sys
from typing import List

import streamlit as st
import importlib
from PIL import Image
import pandas as pd

# Ensure project root is on path when running via Streamlit
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src import config
from src.infer import predict_image, build_infer_transform
from src.model import DocClassifier
from src.utils import load_label_map, get_device
from src import ocr as ocr_mod
importlib.reload(ocr_mod)
# NOTE: Avoid binding extract_fields/extract_fields_from_pil at import time so that
# Streamlit reruns pick up the latest function signature (include_full_text, etc.).
# We provide small helpers that introspect the live signature each call for robust
# backward compatibility with any cached old version still resident in memory.
import inspect

def _call_extract_fields_from_pil(pil_img, include_full_text: bool):
    import importlib
    from src import ocr as _ocr
    _ocr = importlib.reload(_ocr)  # force reload to invalidate any stale cached def
    fn = _ocr.extract_fields_from_pil
    params = inspect.signature(fn).parameters
    if 'include_full_text' in params:
        return fn(pil_img, include_full_text=include_full_text)
    # Fallback: old signature without kwarg
    return fn(pil_img)

def _call_extract_fields(path: str, include_full_text: bool):
    import importlib
    from src import ocr as _ocr
    _ocr = importlib.reload(_ocr)
    fn = _ocr.extract_fields
    params = inspect.signature(fn).parameters
    if 'include_full_text' in params:
        return fn(path, include_full_text=include_full_text)
    return fn(path)
import torch
import torch.nn.functional as F

APP_NAME = "FinSight AI"
st.set_page_config(page_title=APP_NAME, page_icon="💼", layout="wide")

# Minimal pro theme via CSS
st.markdown(
        """
        <style>
        :root {
            --primary: #0E7490; /* teal-700 */
            --accent: #22D3EE;  /* cyan-400 */
        }
        .stApp { background: linear-gradient(180deg,#0b1220 0%,#0f172a 40%,#111827 100%); color: #e5e7eb; }
        .stMarkdown, .stText, .stDataFrame { color: #e5e7eb !important; }
        .metric-label, .metric-value { color: #e5e7eb !important; }
        .css-1v3fvcr, .css-12oz5g7 { color: #e5e7eb !important; }
        .big-title { font-size: 2rem; font-weight: 700; color: #e5e7eb; }
        .subtle { color: #94a3b8; }
        </style>
        """,
        unsafe_allow_html=True,
)

@st.cache_resource
def load_model_and_labels():
    device = get_device()
    label_map_path = os.path.join(config.MODELS_DIR, "label_map.json")
    id2label = load_label_map(label_map_path)
    classes = [id2label[i] for i in sorted(id2label)]

    weights = os.path.join(config.MODELS_DIR, "best.pt")
    ckpt = torch.load(weights, map_location=device)
    model = DocClassifier(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, classes, device


def classify_image(model, device, classes, pil_img: Image.Image):
    tfm = build_infer_transform(config.IMG_SIZE)
    import numpy as np
    arr = np.array(pil_img.convert("RGB"))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    import torch
    x = tfm(image=arr)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    top_idx = probs.argmax()
    return classes[top_idx], float(probs[top_idx]), probs


def append_ledger(row: dict):
    out_csv = os.path.join(config.OUTPUTS_DIR, "ledger.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = pd.DataFrame([row])
    if os.path.exists(out_csv):
        df.to_csv(out_csv, mode='a', header=False, index=False)
    else:
        df.to_csv(out_csv, index=False)


st.markdown(f"<div class='big-title'>{APP_NAME} · Financial Document Intelligence Platform</div>", unsafe_allow_html=True)
st.caption("End-to-end classification + OCR · Privacy-by-design · Reproducible MLflow tracking")

with st.sidebar:
    st.header("Controls")
    st.markdown("Upload a scanned document; get class + OCR fields.")
    privacy = st.toggle("Privacy mode (do not save uploads)", value=True, help="Process in-memory only; no temp files or PII artifacts.")
    demo = st.checkbox("Disable OCR (demo)", value=False)
    ocr_backend = st.selectbox("OCR backend", ["easyocr", "tesseract", "paddle"], index=0)
    include_full_text = st.toggle("Include full OCR text", value=True, help="If off, only key fields (amount/date/invoice_no) are returned.")
    # Propagate backend selection so OCR module can rebuild reader if changed
    os.environ["OCR_BACKEND"] = ocr_backend
    config.OCR_BACKEND = ocr_backend  # keep config in sync for any downstream references
    st.markdown("---")
    # Display latest run metadata if present
    run_info_path = os.path.join(config.OUTPUTS_DIR, "run_info.json")
    if os.path.exists(run_info_path):
        with open(run_info_path, "r", encoding="utf-8") as f:
            run_info = json.load(f)
        st.markdown("### Current Model")
        st.markdown(f"**Run:** {run_info.get('run_name','-')}  ")
        if run_info.get("git_commit"):
            st.markdown(f"**Commit:** `{run_info['git_commit'][:7]}`  ")
        st.markdown(f"**Classes:** {run_info.get('n_classes')}  · **Backbone:** {run_info.get('backbone')}  ")
        st.markdown(f"**Val F1:** {run_info.get('best_val_f1'):.3f} · **Test F1:** {run_info.get('test_f1'):.3f}")
    st.caption("Tracked with MLflow (local ./mlruns). Open MLflow UI: mlflow ui --backend-store-uri mlruns")

model, classes, device = load_model_and_labels()

tab1, tab2 = st.tabs(["Classify", "Evaluation"])

with tab1:
    uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, help="Images are processed locally. In privacy mode, nothing is written to disk.")

    # Sample gallery removed per request (kept logic placeholder for future)

    if uploaded_files:
        for f in uploaded_files:
            cols = st.columns([1, 1])
            with cols[0]:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
                st.image(img, caption=f.name, use_container_width=True)
            with cols[1]:
                label, conf, probs = classify_image(model, device, classes, img)
                st.subheader(f"Prediction: {label} ({conf:.2%})")
                if not demo:
                    if privacy:
                        # Always call through compatibility helper
                        fields = _call_extract_fields_from_pil(img, include_full_text=include_full_text)
                    else:
                        tmp_path = os.path.join(config.OUTPUTS_DIR, "tmp", f.name)
                        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
                        img.save(tmp_path)
                        fields = _call_extract_fields(tmp_path, include_full_text=include_full_text)
                    st.json(fields)
                    if st.button(f"Add to ledger: {f.name}"):
                        row = {**fields, "doc_type": label, "confidence": conf, "filename": f.name}
                        append_ledger(row)
                        st.success("Saved to ledger.csv")

with tab2:
    st.subheader("Evaluation Report")
    metrics_path = os.path.join(config.OUTPUTS_DIR, "metrics.json")
    cm_val = os.path.join(config.OUTPUTS_DIR, "confusion_matrix_val.png")
    cm_test = os.path.join(config.OUTPUTS_DIR, "confusion_matrix_test.png")
    curves = os.path.join(config.OUTPUTS_DIR, "training_curves.png")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        cols = st.columns(3)
        cols[0].metric("Val Acc", f"{metrics['val']['accuracy']:.3f}")
        cols[1].metric("Val F1", f"{metrics['val']['f1_macro']:.3f}")
        cols[2].metric("Test F1", f"{metrics['test']['f1_macro']:.3f}")
        st.markdown("### Per-class F1 (Test)")
        rep = metrics.get('test', {}).get('report', {})
        rows = []
        for k, v in rep.items():
            if k in classes and isinstance(v, dict):
                rows.append({"class": k, "f1": v.get("f1-score", 0.0), "precision": v.get("precision", 0.0), "recall": v.get("recall", 0.0)})
        if rows:
            st.dataframe(pd.DataFrame(rows))
        if os.path.exists(curves):
            st.markdown("### Training Curves")
            st.image(curves, caption="Loss, Accuracy, F1 over epochs", use_container_width=True)
    if os.path.exists(cm_val) or os.path.exists(cm_test):
        st.markdown("### Confusion Matrices")
        cols = st.columns(2)
        if os.path.exists(cm_val):
            cols[0].image(cm_val, caption="Validation Confusion Matrix", use_container_width=True)
        if os.path.exists(cm_test):
            cols[1].image(cm_test, caption="Test Confusion Matrix", use_container_width=True)

    st.markdown("---")
    if os.path.exists(os.path.join(config.OUTPUTS_DIR, "ledger.csv")):
        st.subheader("Ledger")
        df = pd.read_csv(os.path.join(config.OUTPUTS_DIR, "ledger.csv"))
        st.dataframe(df.tail(100), use_container_width=True)
        st.download_button("Download ledger.csv", df.to_csv(index=False), file_name="ledger.csv")

    st.markdown("---")
    # Artifact quick downloads
    art_cols = st.columns(4)
    metrics_file = os.path.join(config.OUTPUTS_DIR, "metrics.json")
    best_weights = os.path.join(config.MODELS_DIR, "best.pt")
    run_info_file = os.path.join(config.OUTPUTS_DIR, "run_info.json")
    label_map_file = os.path.join(config.MODELS_DIR, "label_map.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            art_cols[0].download_button("metrics.json", f.read(), file_name="metrics.json")
    if os.path.exists(run_info_file):
        with open(run_info_file, "r", encoding="utf-8") as f:
            art_cols[1].download_button("run_info.json", f.read(), file_name="run_info.json")
    if os.path.exists(label_map_file):
        with open(label_map_file, "r", encoding="utf-8") as f:
            art_cols[2].download_button("label_map.json", f.read(), file_name="label_map.json")
    if os.path.exists(best_weights):
        with open(best_weights, "rb") as f:
            art_cols[3].download_button("best.pt", f.read(), file_name="best.pt")

    st.caption("FinSight AI © Manula Fernando · Local processing · PaddleOCR/EasyOCR/Tesseract backends · MLflow tracked")
