import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1) Streamlit Page Settings
# -----------------------------
st.set_page_config(
    page_title="Computer Vision Image Classifier (ResNet18)",
    layout="centered"
)

st.title("Computer Vision Image Classification (CPU Only)")
st.write("This app uses a pre-trained ResNet18 model from PyTorch Torchvision to classify images.")

# -----------------------------
# 2) Force CPU Only
# -----------------------------
device = torch.device("cpu")

# -----------------------------
# 3) Load Model (ResNet18)
# -----------------------------
@st.cache_resource
def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.eval()
    model.to(device)
    return model, weights

model, weights = load_model()

# Get labels from weights
categories = weights.meta["categories"]

# -----------------------------
# 4) Preprocessing Transform
# -----------------------------
preprocess = weights.transforms()

# -----------------------------
# 5) Upload Image UI
# -----------------------------
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, caption="Input Image", use_container_width=True)

    # -----------------------------
    # 6) Convert image -> tensor
    # -----------------------------
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)

    # -----------------------------
    # 7) Model Inference (No Grad)
    # -----------------------------
    with torch.no_grad():
        output = model(input_tensor)

    # -----------------------------
    # 8) Softmax + Top-5
    # -----------------------------
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    top5_results = []
    for i in range(5):
        label = categories[top5_catid[i]]
        prob = float(top5_prob[i]) * 100
        top5_results.append({"Rank": i+1, "Class": label, "Probability (%)": prob})

    df = pd.DataFrame(top5_results)

    st.subheader("Top-5 Predicted Classes")
    st.dataframe(df, use_container_width=True)

    # -----------------------------
    # 9) Bar Chart for Probabilities
    # -----------------------------
    st.subheader("Prediction Probability Bar Chart")

    fig, ax = plt.subplots()
    ax.bar(df["Class"], df["Probability (%)"])
    ax.set_ylabel("Probability (%)")
    ax.set_xlabel("Predicted Class")
    ax.set_title("Top-5 Predictions")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)

    # -----------------------------
    # 10) Explanation / Process Path
    # -----------------------------
    st.markdown("### Process Path (How the model predicts)")
    st.write("""
    1. User uploads an image.
    2. Image is converted into RGB format and resized/cropped based on ResNet18 recommended preprocessing.
    3. Image is converted into a tensor and passed into ResNet18 model (CPU only).
    4. Model outputs raw scores (logits).
    5. Softmax converts logits into probabilities.
    6. Top-5 classes with highest probabilities are displayed.
    7. Bar chart visualizes the predicted probabilities.
    """)