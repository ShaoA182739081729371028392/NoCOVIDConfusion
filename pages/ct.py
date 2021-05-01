import streamlit as st
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
sys.path.append('../')
import deep_learning.ct_model as ct_model
SAMPLE_IMAGES = 'https://github.com/ShaoA182739081729371028392/NoMoreCOVIDConfusion/tree/main/Sample%20CT%20Scans'
model = ct_model.load_model()
def write():
    st.title("COVID-19 Classification from CT Scans.")
    st.write("EfficientNetB0 + Vision Transformer Model to classify COVID-19 in Lung CT Scans with 99% Accuracy.")
    st.write(f"Feel Free to Download Sample CT Scans [Here]({SAMPLE_IMAGES}). Images are 256x256 and are grayscale.")
    st.write("Prediction/Inference is lightning quick due to EfficientNet Architecture and Efficient-Performer Attention in Vision Transformer.")
    files_uploaded = st.file_uploader("Please Upload a PNG or JPG File of a CT scan.", type = ['png', 'jpeg', 'jpg'])
    if files_uploaded is not None:
        image = np.array(Image.open(io.BytesIO(files_uploaded.read())))[:, :, 0:3]
        st.image(image)
        predicted = ct_model.predict(model, image)
        st.write(f"Predicted: **{'COVID' if predicted else 'NO COVID'}**")