from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

CLASS_LABELS = [
    "akiec",   # Actinic keratosis
    "bcc",     # Basal cell carcinoma
    "bkl",     # Benign keratosis
    "df",      # Dermatofibroma
    "mel",     # Melanoma
    "nv",      # Melanocytic nevi
    "vasc",    # Vascular lesions
]

CLASS_NAMES = {
    "akiec": "Actinic Keratosis (Pre-cancerous)",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma (Benign)",
    "mel": "Melanoma (Malignant)",
    "nv": "Melanocytic Nevi (Benign Mole)",
    "vasc": "Vascular Lesion (Benign)",
}

MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}

BACKBONE = "MobileNetV2"
