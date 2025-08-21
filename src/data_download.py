import argparse
import os
import shutil
from typing import List, Optional

from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont

from . import config

SUPPORTED = {"invoice", "receipt"}
KAGGLE_SUPPORTED = {"bank_statement", "cheque"}


def save_image(pil_img: Image.Image, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pil_img.convert("RGB").save(out_path, format="JPEG", quality=90)


def download_invoices(per_class: int) -> List[str]:
    # RVL-CDIP does not provide invoice-only splits via HF directly.
    # As a proxy, we pull a small sample from doc-collections with invoice-like images via PubLayNet class or sample local.
    # Here we lean on the 'cord-19' or 'doc' datasets are not invoices; instead, we will fetch from 'denadai/Invoices-Receipts-OCR' if available.
    try:
        ds = load_dataset("denadai/Invoices-Receipts-OCR", split=f"train[:{per_class}]")
    except Exception:
        ds = None

    saved = []
    if ds is not None:
        for i, row in enumerate(ds):
            img = row.get("image")
            if isinstance(img, Image.Image):
                out_path = os.path.join(config.RAW_DIR, "invoice", f"invoice_{i:06d}.jpg")
                save_image(img, out_path)
                saved.append(out_path)
    # Fallback synthetic generation
    if len(saved) < per_class:
        remaining = per_class - len(saved)
        for j in range(remaining):
            img = Image.new("RGB", (900, 1200), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            # Use default font
            lines = [
                "INVOICE",
                f"Invoice No: INV-{j:05d}",
                "Date: 2025-08-21",
                "Vendor: ABC SUPPLIES",
                "Total: LKR 12,345.67",
            ]
            y = 50
            for line in lines:
                d.text((50, y), line, fill=(0, 0, 0))
                y += 60
            out_path = os.path.join(config.RAW_DIR, "invoice", f"invoice_synth_{j:06d}.jpg")
            save_image(img, out_path)
            saved.append(out_path)
    return saved[:per_class]


essentials_receipt_sources = [
    ("cord-v2", "anarasaurus/cord-v2"),  # community mirror if available
    ("cord", "naver-clova-ix/cord-v1"),   # CORD v1 official HF mirror if available
]


def download_receipts(per_class: int) -> List[str]:
    saved: List[str] = []
    for name, repo in essentials_receipt_sources:
        try:
            ds = load_dataset(repo, split=f"train[:{per_class}]")
        except Exception:
            ds = None
        if ds is None:
            continue
        for i, row in enumerate(ds):
            # CORD layouts vary; some provide file paths instead of PIL
            img = row.get("image")
            if isinstance(img, Image.Image):
                out_path = os.path.join(config.RAW_DIR, "receipt", f"receipt_{name}_{i:06d}.jpg")
                save_image(img, out_path)
                saved.append(out_path)
            else:
                img_path = row.get("image_path") or row.get("file_name")
                if img_path and os.path.exists(img_path):
                    out_path = os.path.join(config.RAW_DIR, "receipt", f"receipt_{name}_{i:06d}.jpg")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    shutil.copy(img_path, out_path)
                    saved.append(out_path)
        if len(saved) >= per_class:
            break
    if len(saved) < per_class:
        remaining = per_class - len(saved)
        for j in range(remaining):
            img = Image.new("RGB", (900, 1200), color=(255, 255, 255))
            d = ImageDraw.Draw(img)
            lines = [
                "RECEIPT",
                f"Bill # RCPT-{j:05d}",
                "Date: 21/08/2025",
                "Store: XYZ MART",
                "Total: Rs. 2,345.00",
            ]
            y = 50
            for line in lines:
                d.text((50, y), line, fill=(0, 0, 0))
                y += 60
            out_path = os.path.join(config.RAW_DIR, "receipt", f"receipt_synth_{j:06d}.jpg")
            save_image(img, out_path)
            saved.append(out_path)
    return saved[:per_class]


def _has_kaggle_cli() -> bool:
    from shutil import which
    return which("kaggle") is not None


def kaggle_download(dataset_slug: str, out_dir: str) -> bool:
    """
    Uses kaggle CLI to download and unzip a dataset into out_dir.
    Returns True on success.
    Requires user to have kaggle.json configured.
    """
    import subprocess
    os.makedirs(out_dir, exist_ok=True)
    try:
        # download zip
        subprocess.check_call(["kaggle", "datasets", "download", "-d", dataset_slug, "-p", out_dir, "--force"])  # noqa: S603,S607
        # unzip everything in out_dir
        import zipfile
        for fname in os.listdir(out_dir):
            if fname.lower().endswith(".zip"):
                with zipfile.ZipFile(os.path.join(out_dir, fname), 'r') as zf:
                    zf.extractall(out_dir)
                os.remove(os.path.join(out_dir, fname))
        return True
    except Exception as e:
        print(f"[warn] Kaggle download failed for {dataset_slug}: {e}")
        return False


def materialize_images_from_folder(src_dir: str, dst_dir: str, prefix: str, limit: int) -> List[str]:
    saved: List[str] = []
    if not os.path.isdir(src_dir):
        return saved
    os.makedirs(dst_dir, exist_ok=True)
    for i, fname in enumerate(os.listdir(src_dir)):
        if i >= limit:
            break
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, f"{prefix}_{i:06d}.jpg")
            try:
                img = Image.open(src).convert("RGB")
                save_image(img, dst)
                saved.append(dst)
            except Exception:
                continue
    return saved


def download_kaggle_bank_statements(per_class: int, dataset_slug: Optional[str] = None) -> List[str]:
    """
    Try downloading a bank statements dataset from Kaggle into a temporary folder, then
    copy images into data/raw/bank_statement/ up to per_class.
    """
    dataset_slug = dataset_slug or "shantanudhakadd/bank-statement-images"  # example; change as needed
    target_class_dir = os.path.join(config.RAW_DIR, "bank_statement")
    os.makedirs(target_class_dir, exist_ok=True)
    if _has_kaggle_cli():
        tmp_dir = os.path.join(config.DATA_DIR, "kaggle_cache", "bank_statement")
        if kaggle_download(dataset_slug, tmp_dir):
            # find images
            saved = materialize_images_from_folder(tmp_dir, target_class_dir, "bank_statement", per_class)
            if saved:
                return saved
    # fallback synthetic
    saved: List[str] = []
    for j in range(per_class):
        img = Image.new("RGB", (900, 1200), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        lines = [
            "BANK STATEMENT",
            f"Account: 1234-{j:04d}",
            "Date: 2025-08-21",
            "Balance: LKR 123,456.78",
        ]
        y = 50
        for line in lines:
            d.text((50, y), line, fill=(0, 0, 0))
            y += 60
        out_path = os.path.join(target_class_dir, f"bank_statement_synth_{j:06d}.jpg")
        save_image(img, out_path)
        saved.append(out_path)
    return saved


def download_kaggle_cheques(per_class: int, dataset_slug: Optional[str] = None) -> List[str]:
    dataset_slug = dataset_slug or "rajeevvijayakumar/cheque-dataset"  # example; change as needed
    target_class_dir = os.path.join(config.RAW_DIR, "cheque")
    os.makedirs(target_class_dir, exist_ok=True)
    if _has_kaggle_cli():
        tmp_dir = os.path.join(config.DATA_DIR, "kaggle_cache", "cheque")
        if kaggle_download(dataset_slug, tmp_dir):
            saved = materialize_images_from_folder(tmp_dir, target_class_dir, "cheque", per_class)
            if saved:
                return saved
    # fallback synthetic
    saved: List[str] = []
    for j in range(per_class):
        img = Image.new("RGB", (1200, 600), color=(250, 250, 245))
        d = ImageDraw.Draw(img)
        lines = [
            "CHEQUE",
            f"Cheque No: CHQ-{j:06d}",
            "Pay: JOHN DOE",
            "Amount: Rs. 9,876.54",
            "Date: 21/08/2025",
        ]
        y = 40
        for line in lines:
            d.text((60, y), line, fill=(0, 0, 0))
            y += 40
        out_path = os.path.join(target_class_dir, f"cheque_synth_{j:06d}.jpg")
        save_image(img, out_path)
        saved.append(out_path)
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_class", type=int, default=500)
    parser.add_argument("--classes", nargs="+", default=list(SUPPORTED | KAGGLE_SUPPORTED))
    parser.add_argument("--kaggle_bank_dataset", type=str, default=None,
                        help="Optional Kaggle dataset slug for bank statements")
    parser.add_argument("--kaggle_cheque_dataset", type=str, default=None,
                        help="Optional Kaggle dataset slug for cheques")
    args = parser.parse_args()

    os.makedirs(config.RAW_DIR, exist_ok=True)

    classes = [c.lower() for c in args.classes]
    for c in classes:
        if c not in (SUPPORTED | KAGGLE_SUPPORTED):
            print(f"[warn] Unsupported class '{c}'. Supported: {SUPPORTED | KAGGLE_SUPPORTED}")

    if "invoice" in classes:
        inv = download_invoices(args.per_class)
        print(f"Saved invoices: {len(inv)}")
    if "receipt" in classes:
        rec = download_receipts(args.per_class)
        print(f"Saved receipts: {len(rec)}")
    if "bank_statement" in classes:
        bs = download_kaggle_bank_statements(args.per_class, args.kaggle_bank_dataset)
        print(f"Saved bank statements: {len(bs)}")
    if "cheque" in classes:
        ch = download_kaggle_cheques(args.per_class, args.kaggle_cheque_dataset)
        print(f"Saved cheques: {len(ch)}")

    print("Done.")


if __name__ == "__main__":
    main()
