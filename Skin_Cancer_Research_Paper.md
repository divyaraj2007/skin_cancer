# Skin Cancer Classification Using Transfer Learning on HAM10000 Dermoscopic Images

**Date:** August 29, 2026  
**Model:** MobileNetV2 transfer learning  
**Dataset:** Balanced HAM10000/ISIC sample, 280 real dermoscopic images  
**Classes:** `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`

## Abstract

This paper presents a machine learning experiment for multi-class skin lesion classification using real dermoscopic images from the HAM10000/ISIC dataset. A MobileNetV2 convolutional neural network pre-trained on ImageNet was adapted for seven lesion categories: actinic keratosis (`akiec`), basal cell carcinoma (`bcc`), benign keratosis (`bkl`), dermatofibroma (`df`), melanoma (`mel`), melanocytic nevus (`nv`), and vascular lesion (`vasc`). The processed dataset contains 280 real images, balanced to 40 images per class. The model was trained for 10 epochs with an 80/20 train-validation split. On the 56-image validation set, the model achieved 0.5000 accuracy, 0.4305 weighted precision, 0.5000 weighted recall, and 0.4501 weighted F1-score. Error analysis showed that dermatofibroma was the hardest class to distinguish, especially from basal cell carcinoma and melanoma. The results demonstrate that the pipeline is functional but also show the limitations of training a deep learning classifier on a very small real-image sample.

## 1. Introduction

Skin cancer is one of the most common forms of cancer worldwide, and early detection is essential for reducing morbidity and mortality. Dermoscopy improves visual inspection by revealing subsurface lesion structures, but interpretation requires clinical expertise. Machine learning, especially convolutional neural networks (CNNs), has become an important research direction for supporting lesion classification from dermoscopic images.

The objective of this project is to build and evaluate a reproducible skin lesion classification pipeline using real HAM10000/ISIC images. The project focuses on seven common lesion categories and reports actual training metrics, confusion matrix findings, and visual performance summaries.

## 2. Literature Review

Deep learning has shown strong potential for skin lesion classification. Esteva et al. (2017) demonstrated dermatologist-level skin cancer classification using a CNN trained on a large clinical image dataset. The study helped establish deep learning as a serious approach for dermatology image analysis.

The HAM10000 dataset introduced by Tschandl, Rosendahl, and Kittler (2018) is widely used because it contains 10,015 dermoscopic images across seven diagnostic categories. Its class structure is used in this project. HAM10000 is valuable for benchmarking, but it is naturally imbalanced, with many more nevus images than rare classes such as dermatofibroma and vascular lesions.

Codella et al. (2018) described the ISIC 2017 lesion analysis challenge, which standardized tasks such as lesion classification and segmentation. This work showed the importance of shared datasets and reproducible evaluation protocols.

Haenssle et al. (2018) compared a CNN with dermatologists for melanoma recognition and found that deep learning systems can perform competitively under controlled experimental conditions. However, such models still require careful validation before clinical use.

Adamson and Smith (2018) warned that dermatology AI can worsen health disparities if models are trained on datasets that underrepresent darker skin tones or diverse clinical settings. This is important because many public dermoscopy datasets are not fully representative of real-world populations.

Together, the literature supports CNN-based lesion classification as a promising research area, while also emphasizing dataset quality, bias, external validation, and transparent reporting.

## 3. Materials and Methods

### 3.1 Dataset

The processed dataset was replaced with real HAM10000/ISIC dermoscopic images. The dataset is balanced for this experiment:

| Class | Meaning | Images |
|-------|---------|-------:|
| `akiec` | Actinic keratosis / intraepithelial carcinoma | 40 |
| `bcc` | Basal cell carcinoma | 40 |
| `bkl` | Benign keratosis-like lesion | 40 |
| `df` | Dermatofibroma | 40 |
| `mel` | Melanoma | 40 |
| `nv` | Melanocytic nevus | 40 |
| `vasc` | Vascular lesion | 40 |
| **Total** |  | **280** |

The images are stored in:

```text
data/processed/
```

The data was downloaded using:

```powershell
python scripts\download_ham10000_samples.py --samples-per-class 40
```

### 3.2 Model Architecture

The classifier uses MobileNetV2 with ImageNet pre-trained weights. The convolutional feature extractor is frozen, and the classification head is replaced with a seven-class output layer:

- Dropout
- Linear layer with 256 units
- ReLU activation
- Dropout
- Linear layer with 7 output classes

This approach reduces training cost and is appropriate for small datasets, although performance is limited when the dataset is very small.

### 3.3 Training Configuration

| Parameter | Value |
|-----------|------:|
| Image size | 224 x 224 |
| Batch size | 32 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Validation split | 20% |
| Training images | 224 |
| Validation images | 56 |
| Device | CPU |

Training command:

```powershell
python -m src.train --epochs 10
```

### 3.4 Evaluation Metrics

The model was evaluated using:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Per-class precision, recall, and F1-score
- Confusion matrix
- Training and validation loss/accuracy curves

## 4. Experimental Results

The experiment was completed on August 29, 2026. The final model file is:

```text
models/skin_cancer_model_20260829_213611.pt
```

### 4.1 Overall Metrics

| Metric | Value |
|--------|------:|
| Accuracy | 0.5000 |
| Weighted precision | 0.4305 |
| Weighted recall | 0.5000 |
| Weighted F1-score | 0.4501 |
| Best validation accuracy | 0.5000 |
| Final validation loss | 1.6326 |

The results show that the model learned useful patterns for some classes but did not generalize strongly across all seven categories. This is expected because the training set contains only 224 images.

### 4.2 Per-Class Results

| Class | Precision | Recall | F1-score | Support |
|-------|----------:|-------:|---------:|--------:|
| `akiec` | 0.4444 | 0.6667 | 0.5333 | 6 |
| `bcc` | 0.4667 | 0.7000 | 0.5600 | 10 |
| `bkl` | 0.5000 | 0.3333 | 0.4000 | 9 |
| `df` | 0.0000 | 0.0000 | 0.0000 | 10 |
| `mel` | 0.3636 | 0.4444 | 0.4000 | 9 |
| `nv` | 1.0000 | 0.8333 | 0.9091 | 6 |
| `vasc` | 0.5000 | 0.8333 | 0.6250 | 6 |

The strongest class in this run was `nv`, with an F1-score of 0.9091. The weakest class was `df`, which had zero recall because the model did not correctly identify any dermatofibroma validation images.

## 5. Visualizations

Three figures were generated during evaluation:

**Figure 1. Training and validation curves**  
File: `results/training_history.png`  
This figure shows training accuracy, validation accuracy, training loss, and validation loss across 10 epochs.

**Figure 2. Confusion matrix**  
File: `results/confusion_matrix.png`  
This figure shows which true classes were predicted correctly and which classes were confused with one another.

**Figure 3. Per-class performance chart**  
File: `results/per_class_metrics.png`  
This figure compares precision, recall, and F1-score for all seven lesion categories.

The raw metric files are:

```text
results/classification_report.json
results/metrics_summary.json
results/training_summary.json
```

## 6. Error Analysis

The confusion matrix shows that several classes were difficult for the model to distinguish:

| Actual class | Predicted as | Count | Error rate within actual class |
|--------------|--------------|------:|-------------------------------:|
| `df` | `bcc` | 5 / 10 | 50.0% |
| `bkl` | `mel` | 3 / 9 | 33.3% |
| `df` | `mel` | 3 / 10 | 30.0% |
| `bkl` | `akiec` | 2 / 9 | 22.2% |
| `mel` | `bcc` | 2 / 9 | 22.2% |
| `mel` | `bkl` | 2 / 9 | 22.2% |
| `bcc` | `vasc` | 2 / 10 | 20.0% |
| `df` | `akiec` | 2 / 10 | 20.0% |

### 6.1 Hardest Class: Dermatofibroma

`df` was the hardest class. Its recall was 0.0000, meaning none of the 10 validation dermatofibroma images were correctly classified. Most `df` mistakes went to `bcc` and `mel`. This may happen because dermatofibroma can share color and structural features with other pigmented or reddish lesions in dermoscopic images.

### 6.2 Confusion Between Benign Keratosis and Melanoma

`bkl` was predicted as `mel` in 3 of 9 validation cases. This is clinically important because benign keratosis-like lesions can visually resemble melanoma. A larger dataset, stronger augmentation, and fine-tuning deeper layers may reduce this confusion.

### 6.3 Melanoma Misclassification

`mel` recall was 0.4444, meaning the model detected 4 of 9 melanoma validation images. Misclassified melanoma images were most often predicted as `bcc` or `bkl`. Since melanoma is clinically high-risk, improving melanoma sensitivity should be a priority before any practical use.

### 6.4 Likely Causes of Error

The main causes are:

- Very small training set: only 224 training images.
- High visual similarity between lesion classes.
- Frozen MobileNetV2 backbone may not adapt enough to dermoscopic features.
- Validation support is small, so each error strongly changes the score.
- No segmentation or lesion-cropping step was used.
- No metadata such as age, sex, or lesion location was used.

## 7. Discussion

The experiment confirms that the project pipeline can train, evaluate, and produce interpretable outputs on real dermoscopic images. However, the performance is not yet suitable for clinical or diagnostic use. Accuracy of 0.5000 is above random chance for seven balanced classes, but the low weighted F1-score and poor `df` recall show that the model is undertrained.

The results are still useful because they identify clear improvement targets. The most important next steps are to train on the full HAM10000 dataset, use class-balanced sampling or loss weighting, fine-tune part of the MobileNetV2 backbone, and add explainability methods such as Grad-CAM.

## 8. Conclusion

This project now includes a complete research workflow: real dataset preparation, model training, metric reporting, visualizations, literature review, and error analysis. The MobileNetV2 classifier reached 0.5000 validation accuracy and 0.4501 weighted F1-score on a small balanced HAM10000 sample. The model performed best on `nv` and `vasc`, while `df`, `bkl`, and `mel` were harder to distinguish. The experiment should be treated as a reproducible baseline, not as a final medical-grade classifier.

## References

Adamson, A. S., & Smith, A. (2018). Machine learning and health care disparities in dermatology. *JAMA Dermatology*, 154(11), 1247-1248. https://doi.org/10.1001/jamadermatol.2018.2348

Codella, N. C. F., Gutman, D., Celebi, M. E., Helba, B., Marchetti, M. A., Dusza, S. W., Kalloo, A., Liopyris, K., Mishra, N., Kittler, H., & Halpern, A. (2018). Skin lesion analysis toward melanoma detection: A challenge at the 2017 International Symposium on Biomedical Imaging. *IEEE International Symposium on Biomedical Imaging*, 168-172. https://doi.org/10.1109/ISBI.2018.8363547

Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 542, 115-118. https://doi.org/10.1038/nature21056

Haenssle, H. A., Fink, C., Schneiderbauer, R., Toberer, F., Buhl, T., Blum, A., Kalloo, A., Hassen, A. B. H., Thomas, L., Enk, A., Uhlmann, L., & Reader Study Groups. (2018). Man against machine: Diagnostic performance of a deep learning convolutional neural network for dermoscopic melanoma recognition in comparison to 58 dermatologists. *Annals of Oncology*, 29(8), 1836-1842. https://doi.org/10.1093/annonc/mdy166

Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161. https://doi.org/10.1038/sdata.2018.161

## Reproducibility Checklist

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Research paper | Complete | This file |
| Experimental results | Complete | Section 4 and `results/training_summary.json` |
| Visualizations | Complete | Section 5 and `results/*.png` |
| Literature review | Complete | Section 2 with citations |
| Error analysis | Complete | Section 6 with confusion-matrix findings |

