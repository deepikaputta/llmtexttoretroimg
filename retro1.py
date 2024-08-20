import streamlit as st
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch

# API Token setup
API_TOKEN = "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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Streamlit app interface
st.title("Artistic Style Transfer with Arcane Diffusion")
st.write("Upload an image to apply an artistic style!")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Load the model
    st.write("Loading the model...")
    model_id = "nitrosocke/Arcane-Diffusion"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_auth_token=API_TOKEN)
   

    # Apply the artistic style
    st.write("Applying the artistic style...")
    result = pipe(prompt="", init_image=image, strength=0.5).images[0]

    # Display the transformed image
    st.image(result, caption='Artistic Styled Image', use_column_width=True)


