from fastai.vision import open_image, load_learner, image
import streamlit as st
import numpy as np
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow
import os
import time

import PIL.Image
import requests
from io import BytesIO

def main():
    # App title
    st.title("Son Tung MTP vs G-Dragon")

    # Image source selection
    option = st.radio('', ['Choose a test image', 'Choose your own image'])

    if option == 'Choose a test image':

        # Test image selection
        test_images = os.listdir('data/test/')
        test_image = st.selectbox(
            'Please select a test image:', test_images)

        # Temporarily displays a message while executing 
        with st.spinner('Wait for it...'):
            time.sleep(1.2)

        # Read the image
        file_path = 'data/test/' + test_image
        img = open_image(file_path)

        # Load model and predict
        model = load_learner('data/train/')
        pred_class = model.predict(img)[0]

        # Draw celebratory balloons
        st.balloons()

        # Display the prediction
        if str(pred_class) == 'mtp':
            st.success("This is Son Tung MTP from Vietnam!")
        else:
            st.success("This is G-Dragon from Korea.")

        # Display the image
        display_img = mpimg.imread(file_path)
        st.image(display_img, use_column_width=True)

    else:
        url = st.text_input("Please input an url:")

        if url =! "":
            # Read image from the url
            response = requests.get(url)
            import_img = PIL.Image.open(BytesIO(response.content))

            # Transform the image to feed into the model
            img = import_img.convert('RGB')
            img = image.pil2tensor(img, np.float32).div_(255)
            img = image.Image(img)

            # Load model and make
            model = load_learner('data/train/')
            pred_class = model.predict(img)[0]

            # Display the prediction
            if str(pred_class) == 'mtp':
                st.success("This is Son Tung MTP from Vietnam!")
            else:
                st.success("This is G-Dragon from Korea.")

            st.image(np.asarray(import_img), use_column_width=True)


if __name__ == "__main__":
    main()