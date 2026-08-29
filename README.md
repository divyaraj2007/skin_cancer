# Skin Cancer ML Project

Machine learning project for skin lesion classification using **MobileNetV2 (PyTorch)** transfer learning and the **HAM10000** class schema (7 classes).

> **Note:** Uses PyTorch (compatible with Python 3.14). TensorFlow is not required.

## Quick Start (Demo Mode)

Works immediately without downloading HAM10000:

```powershell
pip install -r requirements.txt
python -m src.train --demo --epochs 10
python -m src.predict data\processed\mel\mel_000.jpg
streamlit run app.py
```

## Real Image Sample

A real, no-logo dermoscopic sample image has been added for testing:

```text
data/real_samples/nv/ISIC_0024307_nv_real.jpg
```

Source: ISIC Archive / HAM10000, image `ISIC_0024307`. Metadata lists this image as a dermoscopic nevus case. License: CC-BY-NC. Attribution: ViDIR Group, Department of Dermatology, Medical University of Vienna.

Use this command to test prediction on the real sample:

```powershell
python -m src.predict data\real_samples\nv\ISIC_0024307_nv_real.jpg
```

For model training, prefer the full HAM10000 or ISIC dataset instead of the generated demo dataset. The demo images are only for pipeline testing.

## Project Structure

```
skin_cancer_ml/
app.py                  # Streamlit web UI
config.py               # Settings and class labels
requirements.txt
scripts/
  prepare_ham10000.py
src/
  demo_data.py          # Synthetic demo dataset
  data_loader.py
  model.py
  train.py
  predict.py
data/processed/         # Real HAM10000 sample images for training
models/                 # Saved .pt models
results/                # Charts and metrics
```

## Current Processed Dataset

`data/processed` has been replaced with real HAM10000 dermoscopic images, not synthetic/demo images.

| Class | Real images |
|-------|------------:|
| akiec | 40 |
| bcc | 40 |
| bkl | 40 |
| df | 40 |
| mel | 40 |
| nv | 40 |
| vasc | 40 |
| **Total** | **280** |

The images were downloaded using:

```powershell
python scripts\download_ham10000_samples.py --samples-per-class 40
```

The downloader uses HAM10000 metadata labels and ISIC image URLs, then writes the files into class folders compatible with `torchvision.datasets.ImageFolder`.

## Latest Training Results

Run completed on **2026-08-29** using:

```powershell
python -m src.train --demo --epochs 10
```

Important: these metrics are from the earlier generated demo dataset run. Since `data/processed` now contains real HAM10000 images, retrain before reporting final real-image metrics.

| Metric | Value |
|--------|------:|
| Accuracy | 0.8214 |
| Weighted precision | 0.8382 |
| Weighted recall | 0.8214 |
| Weighted F1 | 0.8001 |
| Best validation accuracy | 0.8214 |
| Final validation loss | 0.9976 |

Per-class validation metrics:

| Class | Precision | Recall | F1 | Support |
|-------|----------:|-------:|---:|--------:|
| akiec | 1.0000 | 1.0000 | 1.0000 | 6 |
| bcc | 1.0000 | 1.0000 | 1.0000 | 10 |
| bkl | 0.7273 | 0.8889 | 0.8000 | 9 |
| df | 0.6667 | 0.2000 | 0.3077 | 10 |
| mel | 1.0000 | 1.0000 | 1.0000 | 9 |
| nv | 0.4545 | 0.8333 | 0.5882 | 6 |
| vasc | 1.0000 | 1.0000 | 1.0000 | 6 |

Generated artifacts:

| Artifact | File |
|----------|------|
| Trained model | `models/skin_cancer_model_20260829_204713.pt` |
| Default model copy | `models/skin_cancer_model.pt` |
| Best checkpoint | `models/best_model.pt` |
| Training loss/accuracy curves | `results/training_history.png` |
| Confusion matrix | `results/confusion_matrix.png` |
| Per-class performance chart | `results/per_class_metrics.png` |
| Full classification report | `results/classification_report.json` |
| Metrics summary | `results/metrics_summary.json` |
| Training summary | `results/training_summary.json` |

Classes that were hardest to distinguish in this run:

| Actual class | Predicted as | Count | Rate |
|--------------|--------------|------:|-----:|
| df | nv | 5 / 10 | 50.0% |
| df | bkl | 3 / 10 | 30.0% |
| nv | df | 1 / 6 | 16.7% |
| bkl | nv | 1 / 9 | 11.1% |

The main confusion was around benign-looking synthetic classes: `df` was most often mistaken for `nv` and `bkl`, while `nv` and `bkl` each had a smaller number of mistakes involving nearby benign classes.

## Train on Real HAM10000 Data

1. Download from Harvard Dataverse:
   https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

2. Place files:
   ```
   data/raw/HAM10000_metadata.csv
   data/raw/images/*.jpg
   ```

3. Prepare and train:
   ```powershell
   python scripts/prepare_ham10000.py
   python -m src.train --epochs 25
   ```

## Predict

```powershell
python -m src.predict path\to\lesion.jpg
```

## Classes

| Code | Disease | Risk |
|------|---------|------|
| mel | Melanoma | Malignant |
| bcc | Basal Cell Carcinoma | Malignant |
| akiec | Actinic Keratosis | Pre-cancerous |
| nv | Melanocytic Nevi | Benign |
| bkl | Benign Keratosis | Benign |
| df | Dermatofibroma | Benign |
| vasc | Vascular Lesion | Benign |

## Disclaimer

For **research and educational purposes only**. Not FDA-approved medical software.
