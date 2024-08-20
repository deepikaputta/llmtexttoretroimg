import streamlit as st
from PIL import Image
from diffusers import DiffusionPipeline
import torch

# Streamlit app interface
st.title("Retro Image Transformation with Stable Diffusion")
st.write("Upload an image to apply a retro effect using Stable Diffusion!")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Load the model
    st.write("Loading the model...")
    model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    # Apply the retro effect
    st.write("Applying the retro effect...")

    # Use the pipeline to generate an image with a retro effect
    result = pipe(prompt="retro style", init_image=image, strength=0.5).images[0]

    # Display the transformed image
    st.image(result, caption='Retro Styled Image', use_column_width=True)
