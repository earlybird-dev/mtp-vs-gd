from fastai.vision import *
from fastai.metrics import error_rate
import streamlit as st
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
import requests
from io import BytesIO
import time

st.title("Son Tung MTP vs G-Dragon")

option = st.radio(
    '',
     ['Choose a test image', 'Choose your own image'])


if option == 'Choose a test image':

    test_images = os.listdir('data/test/')
    test_image = st.selectbox(
        'Please select:', test_images)

    with st.spinner('Wait for it...'):
        time.sleep(1.5)

    # Read the image
    file_path = 'data/test/' + test_image
    img = open_image(file_path)

    # Load model and predict
    model = load_learner('data')
    pred_class = model.predict(img)[0]

    # Display the prediction
    if str(pred_class) == 'mtp':
        st.success("This is Son Tung MTP from Vietnam!")
    else:
        st.success("This is G-Dragon from Korea.")

    display_img = mpimg.imread(file_path)
    st.image(display_img, use_column_width=True)

# else:
#     url = st.text_input('Or input an url:')
#     try:
#         response = requests.get(url)
#         img = Image.open(BytesIO(response.content))
#         st.image(img, use_column_width=True)
#     except FileNotFoundError:
#         st.error('File not found.')


