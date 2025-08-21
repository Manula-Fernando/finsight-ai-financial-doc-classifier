from __future__ import annotations
import os
import shutil
import glob
from typing import Dict, List, Tuple

import kagglehub
from PIL import Image
import fitz  # PyMuPDF

from . import config
from .data_download import save_image


# Expanded target classes
TARGET_CLASSES = [
    "bank_statement",
    "salary_slip",
    "income_tax_return",
    "utility_bill",
    "cheque",
    "balance_sheet",
    "cash_flow_statement",
    "income_statement",
    "notes",
    "others",
]


def clean_raw_dir():
    if os.path.isdir(config.RAW_DIR):
        for d in os.listdir(config.RAW_DIR):
            full = os.path.join(config.RAW_DIR, d)
            if os.path.isdir(full):
                shutil.rmtree(full)
    os.makedirs(config.RAW_DIR, exist_ok=True)


def _save_image(img: Image.Image, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(img, out_path)


def _render_pdf(fp: str, dst_class: str, count: int, limit: int | None) -> int:
    try:
        doc = fitz.open(fp)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            out_path = os.path.join(config.RAW_DIR, dst_class, f"{dst_class}_{count:06d}_p{i:02d}.jpg")
            _save_image(img, out_path)
            count += 1
            if limit and count >= limit:
                break
        doc.close()
    except Exception:
        pass
    return count


def collect_images(src_patterns: List[str], dst_class: str, limit: int | None = None) -> int:
    count = 0
    for pat in src_patterns:
        for fp in glob.glob(pat, recursive=True):
            try:
                ext = os.path.splitext(fp)[1].lower()
                if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                    img = Image.open(fp).convert("RGB")
                    out_path = os.path.join(config.RAW_DIR, dst_class, f"{dst_class}_{count:06d}.jpg")
                    _save_image(img, out_path)
                    count += 1
                elif ext == ".pdf":
                    count = _render_pdf(fp, dst_class, count, limit)
                if limit and count >= limit:
                    return count
            except Exception:
                continue
    return count


def ingest_personal_financial_docs(limit_per_class: int | None = None) -> Dict[str, int]:
    # mehaksingal/personal-financial-dataset-for-india
    path = kagglehub.dataset_download("mehaksingal/personal-financial-dataset-for-india")
    stats = {c: 0 for c in TARGET_CLASSES}

    # Heuristics: classify images by folder names where possible
    # Build patterns that match by directory name OR file name tokens
    def dir_patterns(dir_globs: List[str]) -> List[str]:
        pats: List[str] = []
        for dg in dir_globs:
            for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.pdf', '.PDF'):
                pats.append(os.path.join(path, '**', dg, f'*{ext}'))
        return pats

    def file_patterns(file_globs: List[str]) -> List[str]:
        pats: List[str] = []
        for fg in file_globs:
            for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.pdf', '.PDF'):
                pats.append(os.path.join(path, '**', f"{fg}{ext}"))
        return pats

    mapping_dir_file: List[Tuple[List[str], List[str], str]] = [
        (# bank statements
            ['*bank*statement*'],
            ['*bank*statement*', '*statement*bank*'],
            'bank_statement'),
        (# salary slips
            ['*salary*slip*', '*payslip*'],
            ['*salary*slip*', '*payslip*'],
            'salary_slip'),
        (# income tax returns / ITR / Form-16
            ['*income*tax*return*', '*itr*', '*form*16*'],
            ['*income*tax*return*', '*itr*', '*form*16*'],
            'income_tax_return'),
        (# utility bills
            ['*utility*'],
            ['*utility*bill*', '*electric*bill*', '*water*bill*', '*gas*bill*'],
            'utility_bill'),
        (# cheques
            ['*check*', '*cheque*'],
            ['*check*', '*cheque*'],
            'cheque'),
    ]

    for dir_globs, file_globs, cls in mapping_dir_file:
        patterns = dir_patterns(dir_globs) + file_patterns(file_globs)
        added = collect_images(patterns, cls, limit=limit_per_class)
        stats[cls] += added
    # Any remaining images not matched could optionally go to 'others' if we had a crawl of all files
    return stats


def ingest_financial_html_datasets(limit_per_class: int | None = None) -> Dict[str, int]:
    # Download datasets primarily containing HTML/text, but they may include PDFs or images we can use.
    roots = []
    roots.append(kagglehub.dataset_download("gopiashokan/financial-document-classification-dataset"))
    roots.append(kagglehub.dataset_download("drcrabkg/financial-statements-clustering"))

    def build_patterns(roots: List[str], tokens: List[str]) -> List[str]:
        pats: List[str] = []
        for r in roots:
            for tok in tokens:
                for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.pdf', '.PDF'):
                    pats.append(os.path.join(r, '**', f"*{tok}*{ext}"))
        return pats

    stats = {c: 0 for c in TARGET_CLASSES}
    # Map corporate financial statements
    corp_mappings: List[Tuple[List[str], str]] = [
        (["balance", "balance-sheet"], "balance_sheet"),
        (["cash", "flow", "statement", "cash-flow"], "cash_flow_statement"),
        (["income", "statement", "profit", "loss", "p&l"], "income_statement"),
        (["notes", "notes-to-accounts", "notes_to_accounts"], "notes"),
    ]
    for tokens, cls in corp_mappings:
        # Join tokens into patterns e.g., '*income*statement*'
        join_patterns = ["*" + "*".join(tokens) + "*"]
        pats = build_patterns(roots, join_patterns)
        added = collect_images(pats, cls, limit=limit_per_class)
        stats[cls] += added

    # Note: We intentionally skip 'others' unless you want a catch-all; can be added by scanning remaining files.
    return stats


def ingest_bcsd_cheques(limit: int | None = None) -> Dict[str, int]:
    path = kagglehub.dataset_download("saifkhichi96/bank-checks-signatures-segmentation-dataset")
    patterns = [
        os.path.join(path, "**", "*.jpg"),
        os.path.join(path, "**", "*.jpeg"),
        os.path.join(path, "**", "*.png"),
    ]
    added = collect_images(patterns, "cheque", limit)
    return {"cheque": added}


def summarize_and_filter_classes(min_images: int = 20) -> List[str]:
    """Return the list of classes present in RAW_DIR with at least min_images, remove tiny/error classes."""
    present: List[str] = []
    for cls in TARGET_CLASSES:
        cls_dir = os.path.join(config.RAW_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        n = len([f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        if n >= min_images:
            present.append(cls)
        else:
            # remove undersized/error classes
            shutil.rmtree(cls_dir, ignore_errors=True)
    return present


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit_per_class", type=int, default=None)
    parser.add_argument("--min_images", type=int, default=20, help="Keep only classes with at least this many images")
    args = parser.parse_args()

    print("Cleaning existing raw data...")
    clean_raw_dir()

    os.makedirs(config.RAW_DIR, exist_ok=True)
    stats_total = {c: 0 for c in TARGET_CLASSES}

    print("Ingesting Personal Financial Docs (India)...")
    s1 = ingest_personal_financial_docs(args.limit_per_class)
    for k, v in s1.items():
        stats_total[k] += v

    print("Ingesting HTML datasets (NLP only)...")
    ingest_financial_html_datasets(args.limit_per_class)

    print("Ingesting BCSD Cheques...")
    s3 = ingest_bcsd_cheques(args.limit_per_class)
    for k, v in s3.items():
        stats_total[k] += v

    print("Collected images per class:", stats_total)
    kept = summarize_and_filter_classes(min_images=args.min_images)
    print("Kept classes (>= min_images):", kept)
    dropped = [c for c in TARGET_CLASSES if c not in kept]
    if dropped:
        print("Dropped classes due to insufficient data:", dropped)


if __name__ == "__main__":
    main()
