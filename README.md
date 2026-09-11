# Skin Cancer ML Project

Machine learning project for skin lesion classification using **MobileNetV2 (PyTorch)** transfer learning and the **HAM10000** class schema (7 classes).

> **Note:** Uses PyTorch (compatible with Python 3.10 through 3.14). TensorFlow is not required.  
> Detailed release specification: See [RELEASE.md](RELEASE.md).

---

## Quick Start (Portable Setup & Demo Mode)

Works immediately on Windows, Linux, and macOS without downloading external datasets:

```powershell
# 1. Navigate to project root
cd skin_cancer_ml

# 2. Set up virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
# source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train on isolated synthetic demo dataset (data/demo/)
python -m src.train --demo --epochs 5

# 5. Predict on demo image with evaluation safeguards
python -m src.predict data/demo/mel/mel_000.jpg

# 6. Launch interactive Streamlit interface
streamlit run app.py
```

---

## Real Image Benchmark Sample

A verified real dermoscopic nevus sample image is included in `data/real_samples/`:

```text
data/real_samples/nv/ISIC_0024307_nv_real.jpg
```

- **Source:** ISIC Archive / HAM10000, image `ISIC_0024307` (ViDIR Group, Medical University of Vienna).
- **License:** CC BY-NC 4.0.

Test prediction with clinical safeguards:

```powershell
python -m src.predict data/real_samples/nv/ISIC_0024307_nv_real.jpg
```

---

## Project Structure & Data Separation

Synthetic test images and real clinical research data are strictly separated:

```
skin_cancer_ml/
├── app.py                      # Streamlit web UI with clinical disclaimer & ABCDE guide
├── config.py                   # Central settings, split ratios, safety thresholds
├── requirements.txt            # Streamlined dependencies (PyTorch + Streamlit)
├── RELEASE.md                  # Comprehensive release specification
├── scripts/
│   ├── download_ham10000_samples.py  # Download real samples with lesion deduplication
│   └── prepare_ham10000.py           # Prepare raw Dataverse files
├── src/
│   ├── demo_data.py            # Generates synthetic data strictly in data/demo/
│   ├── data_loader.py          # Stratified train/val/test splits & anti-leakage guards
│   ├── model.py                # MobileNetV2 architecture & transfer learning
│   ├── train.py                # Training loop, checkpointing, and test evaluation
│   └── predict.py              # Inference with input validation & risk safeguards
├── data/
│   ├── demo/                   # Isolated synthetic demo images
│   ├── processed/              # Real HAM10000 processed sample images
│   ├── raw/                    # Raw HAM10000 metadata CSV
│   └── real_samples/           # Single benchmark cases
├── models/                     # Saved .pt model checkpoints
└── results/                    # Confusion matrix, metrics JSON, history plots
```

---

## Evaluation Safeguards & Protocol

This project enforces rigorous evaluation safeguards to prevent data contamination and overly optimistic reporting:

1. **Strict Data Separation:** Synthetic demo generation writes exclusively to `data/demo/` and will never overwrite or dilute real HAM10000 images in `data/processed/`.
2. **Stratified 3-Way Partitioning:** Dataset is split into 70% Train, 15% Validation (for checkpoint selection), and 15% Holdout Test. Stratification preserves class proportions across all subsets.
3. **Leakage Prevention:** Programmatic assertions verify zero index overlap between partitions (`train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅`).
4. **Lesion Deduplication:** Downloader scripts deduplicate rows by `lesion_id` to prevent multi-image lesions from leaking across partitions.
5. **Model Selection Integrity:** The training pipeline reloads `models/best_model.pt` before running final evaluation on the holdout test set, ensuring that reported test metrics reflect the optimal checkpoint rather than an overfitted final epoch.
6. **Clinical Safety Metrics:** Training evaluation automatically calculates malignant sensitivity, specificity, and false negative rates for high-risk categories (`mel`, `bcc`, `akiec`).
7. **Inference Guards:** Inputs are validated for resolution ($\ge 32\times 32$) and non-blank variance ($\sigma \ge 3.0$). Top predictions with $< 40\%$ confidence are flagged as high uncertainty, and cases with $\ge 15\%$ combined malignant probability trigger clinical safety alerts even if the top predicted class is nominally benign.

---

## Current Processed Dataset

`data/processed` contains 280 real HAM10000 dermoscopic images balanced across all 7 diagnostic classes:

| Class | Diagnostic Category | Nature | Images |
|-------|---------------------|--------|-------:|
| `akiec` | Actinic Keratosis / Bowen's disease | Pre-cancerous | 40 |
| `bcc` | Basal Cell Carcinoma | Malignant | 40 |
| `bkl` | Benign Keratosis (Solar Lentigo / Seborrheic) | Benign | 40 |
| `df` | Dermatofibroma | Benign | 40 |
| `mel` | Melanoma | Malignant | 40 |
| `nv` | Melanocytic Nevi (Common Mole) | Benign | 40 |
| `vasc` | Vascular Lesions (Angioma, Pyogenic Granuloma) | Benign | 40 |
| **Total** | | | **280** |

Download / refresh balanced real images using:
```powershell
python scripts/download_ham10000_samples.py --samples-per-class 40
```

---

## Training on Full HAM10000 Data

1. Download the complete dataset from Harvard Dataverse:
   https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

2. Place raw files into `data/raw/`:
   ```
   data/raw/HAM10000_metadata.csv
   data/raw/images/*.jpg
   ```

3. Prepare and run training:
   ```powershell
   python scripts/prepare_ham10000.py
   python -m src.train --epochs 25
   ```

---

## Clinical Safety, Medical Disclaimer & Ethical Use

> [!WARNING]
> ### Crucial Medical & Clinical Disclaimer
> - **Educational & Research Prototype Only:** This project is intended solely for machine learning research, algorithmic transparency analysis, and educational demonstration.
> - **NOT an FDA-Cleared or CE-Marked Medical Device:** This system has **NOT** undergone clinical trials and has not been cleared or approved by any regulatory health agency.
> - **Never Use for Self-Diagnosis:** This tool **CANNOT** replace professional in-person medical evaluation, dermoscopy, or biopsy by a board-certified dermatologist or qualified healthcare provider.
> - **High Risk of False Negatives:** Early or amelanotic melanomas may appear visually benign to convolutional neural networks. A non-malignant AI output must never be interpreted as an "all clear" or evidence of safety.
> - **Demographic Bias Notice:** Public dermatology datasets (including HAM10000) have well-documented underrepresentation of darker skin tones (Fitzpatrick phototypes IV–VI). Performance on underrepresented groups may be substantially lower.

---

## Legal & Licensing Notice

### Code License
**No open-source code license is granted or invented for this repository.**  
All rights to the source code, training workflows, and documentation are reserved by the author.

### Third-Party Data Attribution
- **HAM10000 Dataset:** Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Sci. Data 5, 180161 (2018). Licensed under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
- **ISIC Archive:** International Skin Imaging Collaboration.
