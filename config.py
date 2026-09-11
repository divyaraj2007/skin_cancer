from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DEMO_DIR = DATA_DIR / "demo"
REAL_SAMPLES_DIR = DATA_DIR / "real_samples"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4

# Data split ratios (stratified)
TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Clinical evaluation & prediction safeguards
MIN_CONFIDENCE_THRESHOLD = 0.40  # Flag predictions with top confidence below 40%
MALIGNANT_RISK_THRESHOLD = 0.15  # Trigger safety alert if combined malignant probability >= 15%

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
