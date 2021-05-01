import streamlit as st
import io
import sys 
sys.path.append('..')
import deep_learning.mask_model as mask_model
from PIL import Image
import numpy as np
model = mask_model.load_model()
SAMPLE_IMAGES = 'https://github.com/ShaoA182739081729371028392/NoMoreCOVIDConfusion/tree/main/Sample%20Mask%20Images'
TARGETS = ['Mouth', 'Chin', 'Nose']
def write():
    st.title('Mask Detection and Correction Suggestion')
    st.write('### Fight the spread of COVID-19 by properly wearing a mask')
    st.write("Uses an EfficientNet + Vision Transformer Architecture to multi-class classify for Exposed Mouth, Nose, or Chin, to ensure proper mask wearing with 97% Accuracy.")
    st.write(f"Feel Free to download sample mask images [here]({SAMPLE_IMAGES}), images are of varying size, some from images of my family, some from the validation dataset.")
    files_uploaded = st.file_uploader("Please input a selfie!", type = ['png', 'jpeg', 'jpg'])
    if files_uploaded is not None:
        image = np.array(Image.open(io.BytesIO(files_uploaded.read())).resize((256, 256)))
        image = np.transpose(image, axes = (1, 0, 2))
        st.image(image)
        pred = mask_model.predict(model, image)
        fails = False
        for target in TARGETS:
            if target not in pred:
                st.write(f"It seems your {target} is uncovered. Please cover it for your safety.")
                fails = True
        if not fails:
            st.write("Nice Going! You\'re ready to head out. Stay Safe!")