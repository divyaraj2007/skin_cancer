"""Streamlit web app for skin cancer ML prediction."""

from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from config import CLASS_LABELS, CLASS_NAMES, IMG_SIZE, MALIGNANT_CLASSES, MODEL_DIR
from src.model import build_model, get_device

st.set_page_config(page_title="Skin Cancer ML Detector", page_icon="🔬", layout="centered")

st.title("Skin Cancer Detection using Machine Learning")
st.caption("MobileNetV2 (PyTorch) | HAM10000 classes | Research & educational use only")

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
    st.error("No trained model found.")
    st.code("cd skin_cancer_ml\npip install -r requirements.txt\npython -m src.train --demo")
    st.stop()

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

uploaded = st.file_uploader("Upload a skin lesion image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]

        ranked = sorted(zip(probs, CLASS_LABELS), key=lambda x: x[0], reverse=True)
        top_prob, top_label = ranked[0]
        top_name = CLASS_NAMES[top_label]
        malignant = top_label in MALIGNANT_CLASSES

        if malignant:
            st.error(f"**{top_name}**")
            st.warning("Possible malignant lesion — consult a dermatologist.")
        else:
            st.success(f"**{top_name}**")
            st.info("Likely benign class — clinical confirmation still recommended.")

        st.metric("Confidence", f"{top_prob * 100:.1f}%")

        st.subheader("All predictions")
        for prob, label in ranked:
            name = CLASS_NAMES[label]
            tag = "Malignant" if label in MALIGNANT_CLASSES else "Benign"
            st.progress(float(prob), text=f"{name} ({tag}): {prob * 100:.1f}%")

st.divider()
st.markdown(
    """
**7 classes:** Melanoma, BCC, Actinic Keratosis, Nevi, Benign Keratosis, Dermatofibroma, Vascular

**Disclaimer:** This tool is for academic demonstration only and is **not** a substitute for professional medical diagnosis.
"""
)
