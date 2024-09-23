import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import pytorch_lightning as pl

from classification.classifiers import LitResNet

# Load your trained model
model = LitResNet.load_from_checkpoint('model.ckpt')
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit app
st.title("Image Classification App")
st.write("Upload an image to classify it into one of the two classes.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img_tensor = transform(image).type(torch.float).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_names = ['Cat!', 'Dog!']
        st.write(f"Prediction: {class_names[predicted.item()]}")
