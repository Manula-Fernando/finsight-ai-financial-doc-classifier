import os
from typing import List

# Default target classes; training can auto-detect available classes from data if not specified
CLASS_NAMES: List[str] = [
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

# Image size for training/inference
IMG_SIZE = 224

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
MLRUNS_DIR = os.path.join(ROOT_DIR, "mlruns")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

# Training defaults
BATCH_SIZE = 32
EPOCHS = 10
LR = 2e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2

BACKBONE = "tf_efficientnet_b0.ns_jft_in1k"  # stronger backbone
RANDOM_STATE = 42
NUM_WORKERS = 2

# MLflow experiment
MLFLOW_EXPERIMENT = "finance-doc-classifier"

# OCR backend: "easyocr" or "tesseract"
OCR_BACKEND = os.environ.get("OCR_BACKEND", "easyocr").lower()
