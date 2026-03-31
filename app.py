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

# Grad-CAM
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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
# TRANSFORM
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
# GRAD-CAM HIGH CONTRAST (Sharp Red)
# ==============================
@st.cache_resource
def get_cam_model(_model, device):
    target_layers = [_model.blocks[-1]]
    cam = GradCAMPlusPlus(model=_model, target_layers=target_layers)
    return cam

def create_high_contrast_cam(visualization):
    gray = cv2.cvtColor(visualization, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray = np.clip((gray - 0.35) / (0.65 + 1e-8), 0, 1)
    gray = np.power(gray, 1.8)
    
    heatmap = cv2.applyColorMap((gray * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    result = (visualization * 0.15 + heatmap * 0.85).astype(np.uint8)
    return result

# ==============================
# PREDICTION + EXPLICABILITÉ
# ==============================
def predict_and_explain(model, image, transform, device, cam, top_k=5):
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        topk = probs.topk(top_k)
    
    idxs = topk.indices.cpu().numpy()
    probs_values = topk.values.cpu().numpy()

    best_idx = int(idxs[0])
    targets = [ClassifierOutputTarget(best_idx)]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    rgb_img = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    base_cam = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return idxs, probs_values, create_high_contrast_cam(base_cam), best_idx, float(probs_values[0])

# ==============================
# DRAW RESULT (style original conservé)
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
st.set_page_config(page_title="SKU Recognition PRO + XAI", layout="wide")
st.title("🛒 Shelf Recognition — Stage 2 (SKU) + Grad-CAM Sharp")

model, device = load_model()
mapping = load_mapping()
sku_info = load_csv()
transform = get_transform()
cam = get_cam_model(model, device)

st.success("✅ Modèle chargé | Grad-CAM Sharp activé")

with st.sidebar:
    st.header("⚙️ Configuration")
    top_k = st.slider("Top-K", 1, 10, 5)
    show_gradcam = st.checkbox("Afficher Grad-CAM Sharp (rouge vif)", value=True)

mode = st.radio("Mode", ["📷 Upload Image", "🎥 Webcam"], horizontal=True)

# ====================== UPLOAD ======================
if mode == "📷 Upload Image":
    uploaded = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_np = np.array(image)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image, caption="Image originale", use_container_width=True)

        with st.spinner("Analyse + Grad-CAM Sharp..."):
            t0 = time.time()
            idxs, probs, gradcam_viz, best_idx, best_prob = predict_and_explain(
                model, image, transform, device, cam, top_k
            )
            latency = (time.time() - t0) * 1000

        sku = mapping.get(best_idx, "Inconnu")

        with col2:
            st.image(draw_result(image_np, sku, best_prob), caption="Prédiction", use_container_width=True)

        if show_gradcam:
            with col3:
                st.image(gradcam_viz, caption="Grad-CAM Sharp (Rouge vif)", use_container_width=True)

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("🎯 SKU", sku)
        c2.metric("📊 Confiance", f"{best_prob:.2%}")
        c3.metric("⚡ Latence", f"{latency:.1f} ms")

        info = sku_info.get(sku, {})
        if info:
            st.subheader("📦 Infos produit")
            st.write(info)

        st.subheader("🔝 Top prédictions")
        for i, (idx, prob) in enumerate(zip(idxs, probs)):
            label = mapping.get(int(idx), "Inconnu")
            st.write(f"{i+1}. {label} → {prob:.2%}")

# ====================== WEBCAM ======================
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

        idxs, probs, gradcam_viz, best_idx, best_prob = predict_and_explain(
            model, pil, transform, device, cam, top_k
        )

        sku = mapping.get(best_idx, "Inconnu")
        frame_out = draw_result(frame_rgb, sku, best_prob)

        if show_gradcam:
            h, w = frame_out.shape[:2]
            gradcam_resized = cv2.resize(gradcam_viz, (w, h))
            combined = np.hstack((frame_out, gradcam_resized))
            frame_placeholder.image(combined, channels="RGB", 
                                   caption=f"Prédiction: {sku} | Grad-CAM Sharp", 
                                   use_container_width=True)
        else:
            frame_placeholder.image(frame_out, channels="RGB", caption=sku, use_container_width=True)

    cap.release()
    st.success("Caméra arrêtée")