import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

# ─── Load Model ─────────────────────────────────────
from torchvision.models import resnet
torch.serialization.add_safe_globals([resnet.ResNet])
model = torch.load("model.pth", map_location=torch.device("cpu"), weights_only=False)
model.eval()

class_names = ['bolt', 'locatingpin', 'nuts', 'washers']

# ─── Setup ──────────────────────────────────────────
st.set_page_config(page_title="🔩 Part Classifier", layout="wide", page_icon="🧠")
st.markdown("<h1 style='text-align: center;'>🔩 AI Mechanical Part Classifier</h1>", unsafe_allow_html=True)
st.caption("Upload an image to identify bolts, nuts, washers, or locating pins.")

# ─── Sidebar: Upload & Chart Only ──────────────────
st.sidebar.title("🔧 Upload Image")
file = st.sidebar.file_uploader("📤 Browse image", type=["jpg", "jpeg", "png"])

if "history" not in st.session_state:
    st.session_state.history = []

# ─── Prediction Logic ───────────────────────────────
if file:
    image = Image.open(file).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)

    with st.spinner("🧠 Classifying..."):
        time.sleep(1)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            conf, pred = torch.max(probs, 0)

        pred_label = class_names[pred]
        conf_score = conf.item() * 100

        st.session_state.history.append({
            "label": pred_label,
            "confidence": conf_score
        })

    # ─── MAIN: Show Image & Prediction Info ──────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("📌 Top Prediction")
        st.success(f"✅ **{pred_label}** ({conf_score:.2f}%)")

        st.markdown("### 🔢 Class Probabilities")
        top4 = sorted(zip(class_names, probs.tolist()), key=lambda x: x[1], reverse=True)
        for name, prob in top4:
            st.write(f"**{name}** — {prob * 100:.2f}%")

# ─── SIDEBAR: Only Chart Below Upload ───────────────
    st.sidebar.markdown("### 📊 Confidence Chart")
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.bar([x[0] for x in top4], [x[1]*100 for x in top4], color='lightblue')
    ax.set_ylabel("Confidence (%)", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title("Per-Class Confidence", fontsize=10)
    plt.xticks(rotation=10)
    st.sidebar.pyplot(fig)

else:
    st.info("📭 Upload an image from the sidebar to begin.")
