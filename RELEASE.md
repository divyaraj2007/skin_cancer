# Skin Cancer ML Project — Release Documentation & Technical Specification

**Release Version:** v0.2.0  
**Framework:** PyTorch (TorchVision MobileNetV2)  
**Task:** 7-Class Skin Lesion Classification (HAM10000 Schema)  
**Intended Audience:** Machine learning researchers, medical computer vision students, algorithmic auditors  

---

## 1. Overview & Release Highlights

This release refactors and hardens the skin lesion classification pipeline to ensure reproducibility, data integrity, rigorous evaluation safeguards, and responsible clinical safety messaging:

- **Portable Installation:** Elimination of machine-specific directory assumptions; standardized dependency specification in `requirements.txt` with PyTorch and Streamlit support.
- **Strict Data Separation:** Segregation of synthetic test data (`data/demo/`) and real dermoscopic images (`data/processed/`), preventing pipeline contamination.
- **Evaluation Safeguards:**
  - Stratified 3-way dataset partitioning (Train / Validation / Holdout Test).
  - Mathematical zero-leakage assertions (`train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅`).
  - Model selection integrity: Reloading `best_model.pt` weights before final evaluation on the held-out test split (preventing optimistic validation bias).
  - Clinical safety metrics: Calculation of sensitivity, specificity, and false-negative rate specifically for malignant categories (`mel`, `bcc`, `akiec`).
  - Input validation: Automatic detection and rejection of low-resolution (<32x32), corrupt, or degenerate/uniform images.
  - Uncertainty & Risk Detection: Automated detection of high uncertainty (<40% top confidence) and non-negligible malignant risk (≥15% combined probability).
- **Safety Messaging Overhaul:** Removal of misleading green "Success" / "All Clear" UX elements for benign predictions; inclusion of prominent medical disclaimers, risk banners, and patient educational guidance (ABCDE melanoma criteria).
- **Legal & Licensing Clarity:** Explicit disclosure that no open-source code license has been granted or invented (author rights reserved), while citing HAM10000 under CC BY-NC 4.0.

---

## 2. Environment & Portable Installation

The codebase requires Python 3.10 through 3.14 on Windows, Linux, or macOS. No GPU is required, although CUDA acceleration is utilized if available.

### 2.1 Clone & Navigate
```bash
# Clone the repository (or navigate to the workspace directory)
cd skin_cancer_ml
```

### 2.2 Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

**On Linux / macOS (bash / zsh):**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2.3 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on Frameworks:** The core training and inference pipeline is implemented purely in **PyTorch**. The `requirements.txt` file installs PyTorch, torchvision, Streamlit, and scientific dependencies. Legacy TensorFlow is not required for the main pipeline.

---

## 3. Data Architecture & Separation Protocol

The dataset directory layout is organized into distinct functional areas to prevent accidental data mixing:

```
data/
├── demo/              # Synthetic lesion images generated for pipeline testing
│   ├── akiec/
│   ├── bcc/
│   ├── bkl/
│   ├── df/
│   ├── mel/
│   ├── nv/
│   └── vasc/
├── processed/         # Real HAM10000 dermoscopic images for model training
│   ├── akiec/ (40 real images)
│   ├── bcc/   (40 real images)
│   ├── bkl/   (40 real images)
│   ├── df/    (40 real images)
│   ├── mel/   (40 real images)
│   ├── nv/    (40 real images)
│   └── vasc/  (40 real images)
├── raw/               # Downloaded HAM10000_metadata.csv and raw archives
└── real_samples/      # Isolated benchmark cases (e.g. nv/ISIC_0024307_nv_real.jpg)
```

### Separation Rules:
1. **Isolated Demo Generation:** Running `python -m src.demo_data` or `python -m src.train --demo` writes exclusively to `data/demo/`. It will never overwrite, alter, or mix files inside `data/processed/`.
2. **Safe Fallback Handling:** If `data/processed/` is empty or missing, `src.data_loader` raises a descriptive `FileNotFoundError` rather than silently generating synthetic shapes into the real data folder.
3. **Lesion Deduplication:** In `scripts/download_ham10000_samples.py`, images are deduplicated by `lesion_id` to safeguard against multi-image lesion leakage across experimental splits.

---

## 4. Evaluation Protocol & Safeguards

In medical image classification, evaluation rigor is essential to prevent false optimism. This project enforces several structural safeguards:

### 4.1 Stratified 3-Way Partitioning
Rather than naive unstratified random splitting, the data loader implements stratified partitioning using `scikit-learn`:
- **Training Partition (70%):** Model parameter optimization via cross-entropy loss.
- **Validation Partition (15%):** Model selection and checkpoint saving (`best_model.pt`).
- **Holdout Test Partition (15%):** Strictly held out from all training iterations, learning rate decisions, and checkpoint criteria.

### 4.2 Leakage Verification
Every loader instantiation executes programmatic assertions:
```python
assert train_set.isdisjoint(val_set)
assert train_set.isdisjoint(test_set)
assert val_set.isdisjoint(test_set)
assert len(train_set) + len(val_set) + len(test_set) == total_samples
```

### 4.3 Unbiased Evaluation Workflow
During training:
1. At each epoch, validation loss and validation accuracy are recorded.
2. The best model checkpoint is saved to `models/best_model.pt`.
3. At the end of training, `best_model.pt` is reloaded into the neural network architecture before running inference on the holdout `test_loader`. This prevents reporting metrics on overfitted or degraded final-epoch weights.

### 4.4 Clinical Safety Metrics
In addition to standard multi-class accuracy and weighted F1-scores, evaluation calculates targeted oncology metrics:
- **Malignant Sensitivity (Recall):** $\frac{\text{True Malignant Detected}}{\text{Total True Malignant}}$
- **Malignant False Negative Rate:** $\frac{\text{Missed Malignant}}{\text{Total True Malignant}}$
- **Benign Specificity:** $\frac{\text{True Benign Identified}}{\text{Total True Benign}}$

### 4.5 Inference Safeguards
The prediction module (`src.predict`) and web UI (`app.py`) enforce runtime guards:
- **Input Integrity:** Validates image file readability, enforces minimum 32x32 pixel dimensions, and detects uniform/blank images via standard deviation checks ($\sigma \ge 3.0$).
- **Uncertainty Guard:** When the highest predicted class probability is $< 40\%$, a prominent high-uncertainty warning is issued.
- **Malignancy Risk Guard:** If the top-ranked class is nominally benign (e.g. Nevus), but the combined probability of malignant classes (`mel` + `bcc` + `akiec`) reaches $\ge 15\%$, an elevated malignancy risk alert is triggered.

---

## 5. Medical Safety, Regulatory Status & Ethical Use

> [!WARNING]
> ### Critical Medical & Regulatory Disclaimer
> 1. **Research Prototype Only:** This software and its associated models are designed strictly for educational demonstration, machine learning benchmarking, and computer vision research.
> 2. **Not a Medical Device:** This application has **NOT** been cleared, approved, certified, or evaluated by the United States Food and Drug Administration (FDA), European Medicines Agency (EMA), or any other national health authority.
> 3. **Non-Diagnostic:** This software is **NOT** a clinical decision support tool and must **NEVER** be used to diagnose, triage, treat, monitor, or manage skin lesions, melanoma, or any other dermatological condition.
> 4. **Do Not Self-Diagnose:** Patients and individuals must never use this tool for personal health assessments or to delay professional consultation. Any new, changing, symptomatic, bleeding, or irregular lesion requires evaluation by a board-certified dermatologist.
> 5. **Known Algorithmic Limitations & False Negatives:**
>    - Experimental computer vision models are vulnerable to false negatives (misclassifying lethal melanomas as harmless nevi or keratoses).
>    - Model performance on small benchmark samples is not generalizable to clinical practice.
>    - **Demographic Bias:** Public dermatology training datasets (including HAM10000) heavily overrepresent lighter skin tones (Fitzpatrick phototypes I–III) and underrepresent deeply pigmented skin (Fitzpatrick phototypes IV–VI). Algorithms trained on such data exhibit substantial performance disparities across diverse populations.
>    - **Domain Shift:** Dermoscopy models trained on polarized dermatoscopic lenses fail when provided standard smartphone or clinical photographs.

---

## 6. Execution Guide

### 6.1 Quickstart (Isolated Demo Pipeline)
Verify the complete pipeline in under two minutes without downloading external datasets:
```powershell
# 1. Generate synthetic images in data/demo/ and train for 5 epochs
python -m src.train --demo --epochs 5

# 2. Run prediction with safeguards on a demo image
python -m src.predict data/demo/mel/mel_000.jpg

# 3. Launch the Streamlit interactive UI
streamlit run app.py
```

### 6.2 Testing on Real Benchmark Sample
```powershell
# Run prediction with safeguards on the real dermoscopic nevus sample
python -m src.predict data/real_samples/nv/ISIC_0024307_nv_real.jpg
```

### 6.3 Full Training on Processed HAM10000 Data
```powershell
# Train for 15 epochs on real processed images (data/processed/)
python -m src.train --epochs 15
```

---

## 7. Legal Notice & Code Licensing

### 7.1 Code Licensing Status
**No open-source code license is granted or implied for this project.**  
All rights to the source code, training scripts, and architectural configurations are reserved by the repository author. No legal open-source license (e.g. MIT, Apache, GPL, BSD) has been applied or invented.

### 7.2 Third-Party Data & Model Attributions
- **HAM10000 Dataset:** Created by ViDIR Group (Medical University of Vienna) and Cliff Rosendahl (University of Queensland). Licensed under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).
- **ISIC Archive:** Images provided by the International Skin Imaging Collaboration (ISIC).
- **MobileNetV2 Weights:** Pretrained ImageNet weights provided via PyTorch `torchvision.models` under the PyTorch Software Foundation license.
