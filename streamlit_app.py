import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

model = torch.load("model.pth", map_location=torch.device("cpu"), weights_only=False)

model.eval()

class_names = ['bolt', 'locatingpin', 'nuts', 'washers']

st.set_page_config(page_title="Bolt Classifier", layout="centered")
st.title("🔩 Mechanical Part Classifier")

file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Input Image", use_container_width=True)


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    st.success(f"Prediction: {class_names[pred]} ({conf.item()*100:.2f}%)")
