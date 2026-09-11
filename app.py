"""Streamlit web app for skin cancer ML prediction with clinical safety safeguards."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from config import (
    CLASS_LABELS,
    CLASS_NAMES,
    IMG_SIZE,
    MALIGNANT_CLASSES,
    MALIGNANT_RISK_THRESHOLD,
    MIN_CONFIDENCE_THRESHOLD,
    MODEL_DIR,
)
from src.model import build_model, get_device
from src.predict import preprocess_image, validate_image_input

st.set_page_config(page_title="Skin Cancer ML Prototype", page_icon="🔬", layout="centered")

st.title("Skin Lesion Classifier (Research Prototype)")
st.caption("MobileNetV2 (PyTorch) | HAM10000 7-Class Schema | Educational & Research Use Only")

# Mandatory Clinical Safety & Regulatory Banner
st.warning(
    """
⚠️ **CRITICAL MEDICAL DISCLAIMER**
- **Non-Diagnostic Tool:** This application is strictly an educational research prototype. It is **NOT** an FDA-cleared, CE-marked, or clinically validated medical device.
- **No Substitute for Clinical Care:** This tool **CANNOT** diagnose cancer or replace an in-person examination, dermoscopy, or biopsy by a board-certified dermatologist.
- **Risk of Errors:** Deep learning models can produce both false positives and false negatives (misclassifying malignancies as benign).
- **Action:** If you are concerned about a skin lesion that is new, changing, itching, or bleeding, consult a qualified physician immediately.
"""
)

MODEL_PATH = MODEL_DIR / "skin_cancer_model.pt"
if not MODEL_PATH.exists():
    MODEL_PATH = MODEL_DIR / "best_model.pt"


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None, None

    device = get_device()
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, device


model, device = get_model()

if model is None:
    st.error("No trained model checkpoint found.")
    st.info(
        "To train a model on the separated demo dataset:\n"
        "```powershell\n"
        "pip install -r requirements.txt\n"
        "python -m src.train --demo\n"
        "```"
    )
    st.stop()

uploaded = st.file_uploader("Upload a dermoscopic lesion image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        image = Image.open(uploaded)
        # Validation safeguard: verify format and integrity
        image.verify()
        image = Image.open(uploaded).convert("RGB")
        width, height = image.size
        if width < 32 or height < 32:
            st.error(f"Image resolution too low ({width}x{height}). Minimum required: 32x32 pixels.")
            st.stop()
        if float(np.std(np.array(image, dtype=np.float32))) < 3.0:
            st.error("Uploaded image lacks visual variation (blank or uniform). Please upload a valid image.")
            st.stop()
    except Exception as exc:
        st.error(f"Invalid image file: {exc}")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        tensor = preprocess_image(image).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]

        ranked = sorted(zip(probs, CLASS_LABELS), key=lambda x: x[0], reverse=True)
        top_prob, top_label = ranked[0]
        top_name = CLASS_NAMES[top_label]
        malignant = top_label in MALIGNANT_CLASSES
        combined_malignant_prob = sum(p for p, lbl in ranked if lbl in MALIGNANT_CLASSES)

        # Evaluation Safeguard: Uncertainty Alert
        if top_prob < MIN_CONFIDENCE_THRESHOLD:
            st.warning(
                f"⚠️ **High Uncertainty**: Top confidence is {top_prob * 100:.1f}%, below the 40% reliability threshold. "
                "The model is uncertain between multiple categories."
            )

        # Safety Messaging: Never use reassuring green success cards for medical predictions
        if malignant:
            st.error(f"**Top Prediction: {top_name}**")
            st.error(
                "🚨 **Potentially Malignant / Pre-cancerous Category**\n\n"
                "Prompt clinical consultation with a dermatologist is strongly recommended."
            )
        elif combined_malignant_prob >= MALIGNANT_RISK_THRESHOLD:
            st.info(f"**Top Prediction: {top_name}**")
            st.warning(
                f"⚠️ **Elevated Malignancy Probability ({combined_malignant_prob * 100:.1f}%)**\n\n"
                "While the highest individual class is benign, the model detects notable probability "
                "for malignant categories. Clinical verification is advised."
            )
        else:
            st.info(f"**Top Prediction: {top_name}**")
            st.caption(
                "ℹ️ *Category is nominally benign. AI predictions cannot rule out cancer; false negatives "
                "occur in computer vision models. Seek professional dermatological evaluation if concerned.*"
            )

        st.metric("Top Confidence", f"{top_prob * 100:.1f}%")
        st.metric("Combined Malignancy Probability", f"{combined_malignant_prob * 100:.1f}%")

        st.subheader("All Class Probabilities")
        for prob, label in ranked:
            name = CLASS_NAMES[label]
            tag = "Malignant" if label in MALIGNANT_CLASSES else "Benign"
            st.progress(float(prob), text=f"{name} ({tag}): {prob * 100:.1f}%")

st.divider()

with st.expander("ℹ️ Clinical Awareness: The ABCDE Guide for Melanoma"):
    st.markdown(
        """
Dermatologists often recommend monitoring pigmented lesions with the **ABCDE** criteria:
- **A — Asymmetry:** One half of the spot does not match the other half.
- **B — Border:** Edges are irregular, ragged, notched, or blurred.
- **C — Color:** Color is not uniform; shades of brown, black, pink, red, white, or blue may be present.
- **D — Diameter:** Spot is larger than 6 mm (about the size of a pencil eraser), though melanomas can be smaller.
- **E — Evolving:** The mole or spot is changing in size, shape, color, or elevation, or symptoms like itching or bleeding develop.

*When in doubt, always have the lesion examined by a medical professional.*
"""
    )

st.caption(
    "HAM10000 7 Diagnostic Classes: Actinic Keratoses (akiec), Basal Cell Carcinoma (bcc), "
    "Benign Keratosis (bkl), Dermatofibroma (df), Melanoma (mel), Melanocytic Nevi (nv), Vascular Lesions (vasc)."
)
