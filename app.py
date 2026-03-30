import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import timm
import os
import time
import pandas as pd
import json

# ==============================
# CONFIG
# ==============================
IMG_SIZE = 224
MODEL_NAME = "mobilenetv3_small_100"
NUM_CLASSES = 113

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "mobilenetv3_small_best.pth")
CSV_PATH = os.path.join(BASE_DIR, "sku_catalog.csv")
MAPPING_PATH = os.path.join(BASE_DIR, "idx_to_class.json")

# ==============================
# TRANSFORM (KAGGLE EXACT)
# ==============================
def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

# ==============================
# LOADS
# ==============================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device

@st.cache_data
def load_mapping():
    with open(MAPPING_PATH, "r") as f:
        return {int(k): v for k, v in json.load(f).items()}

@st.cache_data
def load_csv():
    df = pd.read_csv(CSV_PATH)
    return df.set_index("sku_id").to_dict("index")

# ==============================
# PREDICTION
# ==============================
def predict(model, image, transform, device, top_k=5):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0]
        topk = probs.topk(top_k)
    return topk.indices.cpu().numpy(), topk.values.cpu().numpy()

# ==============================
# DRAW RESULT (STYLE STAGE 1)
# ==============================
def draw_result(image_np, label, conf):
    img = image_np.copy()
    h, w = img.shape[:2]

    color = (0, 255, 0)
    cv2.rectangle(img, (20, 20), (w-20, h-20), color, 6)

    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)

    cv2.rectangle(img, (30, 30), (30 + tw + 20, 30 + th + 20), color, -1)
    cv2.putText(img, text, (40, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 4)
    cv2.putText(img, text, (40, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

    return img

# ==============================
# UI
# ==============================
st.set_page_config(page_title="SKU Recognition PRO", layout="wide")
st.title("🛒 Shelf Recognition — Stage 2 (SKU)")

model, device = load_model()
mapping = load_mapping()
sku_info = load_csv()
transform = get_transform()

st.success("✅ Modèle chargé (MobileNetV3-Small)")

# Sidebar
with st.sidebar:
    st.header("⚙️ Config")
    top_k = st.slider("Top-K", 1, 10, 5)
    conf_threshold = st.slider("Seuil confiance", 0.1, 0.9, 0.3)

# Mode
mode = st.radio("Mode", ["📷 Upload", "🎥 Webcam"], horizontal=True)

# ==============================
# IMAGE MODE
# ==============================
if mode == "📷 Upload":
    uploaded = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_np = np.array(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, use_container_width=True)

        with st.spinner("🔍 Analyse..."):
            t0 = time.time()
            idxs, probs = predict(model, image, transform, device, top_k)
            latency = (time.time() - t0) * 1000

        best_idx = int(idxs[0])
        best_prob = float(probs[0])
        sku = mapping[best_idx]

        img_draw = draw_result(image_np, sku, best_prob)

        with col2:
            st.image(img_draw, use_container_width=True)

        # Metrics
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("🎯 SKU", sku)
        c2.metric("📊 Confiance", f"{best_prob:.2%}")
        c3.metric("⚡ Latence", f"{latency:.1f} ms")

        # Infos produit
        info = sku_info.get(sku, {})
        if info:
            st.subheader("📦 Infos produit")
            st.write(info)

        # Top-K
        st.subheader("🔝 Top prédictions")
        for i, (idx, prob) in enumerate(zip(idxs, probs)):
            st.write(f"{i+1}. {mapping[int(idx)]} → {prob:.2%}")

# ==============================
# WEBCAM MODE
# ==============================
elif mode == "🎥 Webcam":
    st.info("📹 Webcam en temps réel")

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop = st.button("⏹️ Stop")

    while not stop:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)

        idxs, probs = predict(model, pil, transform, device, top_k)

        best_idx = int(idxs[0])
        best_prob = float(probs[0])
        sku = mapping[best_idx]

        frame_out = draw_result(frame_rgb, sku, best_prob)

        frame_placeholder.image(frame_out, use_container_width=True)

    cap.release()
    st.success("Caméra arrêtée")