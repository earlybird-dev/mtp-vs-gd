from fastai.vision import open_image, load_learner, image, torch
import streamlit as st
import numpy as np
import matplotlib.image as mpimg
import os
import time
import PIL.Image
import requests
from io import BytesIO

# App title
st.title("Son Tung MTP vs G-Dragon")

def predict(img, display_img):

    # Display the test image
    st.image(display_img, use_column_width=True)

    # Temporarily displays a message while executing 
    with st.spinner('Wait for it...'):
        time.sleep(3)

    # Load model and make prediction
    model = load_learner('model/data/train/')
    pred_class = model.predict(img)[0]
    pred_prob = round(torch.max(model.predict(img)[2]).item()*100)
    
    # Display the prediction
    if str(pred_class) == 'mtp':
        st.success("This is Son Tung MTP with the probability of " + str(pred_prob) + '%.')
    else:
        st.success("This is G-Dragon with the probability of " + str(pred_prob) + '%.')
    
# Image source selection
option = st.radio('', ['Choose a test image', 'Choose your own image'])

if option == 'Choose a test image':

    # Test image selection
    test_images = os.listdir('model/data/test/')
    test_image = st.selectbox(
        'Please select a test image:', test_images)

    # Read the image
    file_path = 'model/data/test/' + test_image
    img = open_image(file_path)
    # Get the image to display
    display_img = mpimg.imread(file_path)

    # Predict and display the image
    predict(img, display_img)

else:
    url = st.text_input("Please input a url:")

    if url != "":
        try:
            # Read image from the url
            response = requests.get(url)
            pil_img = PIL.Image.open(BytesIO(response.content))
            display_img = np.asarray(pil_img) # Image to display

            # Transform the image to feed into the model
            img = pil_img.convert('RGB')
            img = image.pil2tensor(img, np.float32).div_(255)
            img = image.Image(img)

            # Predict and display the image
            predict(img, display_img)

        except:
            st.text("Invalid url!")