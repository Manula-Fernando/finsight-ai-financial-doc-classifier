from __future__ import annotations
import re
from typing import Dict, Optional

import cv2
import numpy as np
import os
from . import config
try:
    import easyocr
except Exception:
    easyocr = None
try:
    import pytesseract
except Exception:
    pytesseract = None
try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:
    PaddleOCR = None

_amount_re = re.compile(r"(Rs\.?|LKR)\s?([\d,]+(?:\.\d{2})?)", re.IGNORECASE)
_date_re = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|20\d{2}-\d{2}-\d{2})\b")
_invoice_re = re.compile(r"(invoice|inv|bill)\s*(no\.?|#)?\s*[:\-]?\s*([A-Za-z0-9\-]+)", re.IGNORECASE)

_reader = None
_backend_name = None


def get_reader():
    """Return (and lazily build) the OCR reader for the requested backend.

    The desired backend is resolved in this order:
      1. Environment variable OCR_BACKEND (set by Streamlit UI)
      2. config.OCR_BACKEND default

    If the backend selection changes at runtime, the cached reader is rebuilt.
    """
    global _reader, _backend_name
    desired = os.environ.get("OCR_BACKEND", config.OCR_BACKEND).lower()
    if _reader is None or _backend_name != desired:
        # Force rebuild
        _reader = None
        backend = desired
        if backend == "tesseract" and pytesseract is not None:
            _reader = "tesseract"  # sentinel indicating to use pytesseract each call
            _backend_name = "tesseract"
        elif backend == "paddle" and PaddleOCR is not None:
            _reader = PaddleOCR(lang="en", use_angle_cls=True, show_log=False, use_gpu=False)
            _backend_name = "paddle"
        else:
            # fallback easyocr
            if easyocr is None:
                raise RuntimeError("No OCR backend available (easyocr missing and requested backend not usable).")
            _reader = easyocr.Reader(["en"], gpu=False)
            _backend_name = "easyocr"
    return _reader


def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 75, 75)
    th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return th


def extract_fields(image_path: str, include_full_text: bool = True) -> Dict[str, Optional[str]]:
    img = cv2.imread(image_path)
    if img is None:
        return {"amount": None, "date": None, "invoice_no": None, "text": None}
    prep = preprocess_for_ocr(img)
    reader = get_reader()
    if _backend_name == "tesseract" and pytesseract is not None:
        # Use pytesseract
        from PIL import Image
        pil = Image.fromarray(prep)
        text = pytesseract.image_to_string(pil)
        lines = text.splitlines()
    elif _backend_name == "paddle" and PaddleOCR is not None:
        ocr_res = reader.ocr(prep, cls=True)
        lines = []
        for page in ocr_res:
            for box, (txt, conf) in page:
                if conf >= 0.5:
                    lines.append(txt)
        text = "\n".join(lines)
    else:  # easyocr
        results = reader.readtext(prep)
        lines = [r[1] for r in results]
    if _backend_name != "paddle":  # already set for paddle path
        text = "\n".join(lines)

    amount = None
    m = _amount_re.search(text)
    if m:
        amount = m.group(0)

    date = None
    d = _date_re.search(text)
    if d:
        date = d.group(0)

    inv = None
    iv = _invoice_re.search(text)
    if iv:
        inv = iv.group(3)

    out = {"amount": amount, "date": date, "invoice_no": inv}
    if include_full_text:
        out["text"] = text
    return out


def extract_fields_from_pil(pil_img, include_full_text: bool = True) -> Dict[str, Optional[str]]:
    """Run OCR on an in-memory PIL image without writing to disk (privacy-safe)."""
    import cv2
    import numpy as np
    img_rgb = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    prep = preprocess_for_ocr(img_bgr)
    reader = get_reader()
    if _backend_name == "tesseract" and pytesseract is not None:
        from PIL import Image
        pil = Image.fromarray(prep)
        text = pytesseract.image_to_string(pil)
        lines = text.splitlines()
    elif _backend_name == "paddle" and PaddleOCR is not None:
        ocr_res = reader.ocr(prep, cls=True)
        lines = []
        for page in ocr_res:
            for box, (txt, conf) in page:
                if conf >= 0.5:
                    lines.append(txt)
        text = "\n".join(lines)
    else:  # easyocr
        results = reader.readtext(prep)
        lines = [r[1] for r in results]
    if _backend_name != "paddle":
        text = "\n".join(lines)

    amount = None
    m = _amount_re.search(text)
    if m:
        amount = m.group(0)

    date = None
    d = _date_re.search(text)
    if d:
        date = d.group(0)

    inv = None
    iv = _invoice_re.search(text)
    if iv:
        inv = iv.group(3)

    out = {"amount": amount, "date": date, "invoice_no": inv}
    if include_full_text:
        out["text"] = text
    return out
