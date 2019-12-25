from fastai.vision import open_image, load_learner
import streamlit as st
import matplotlib.image as mpimg
import os
import time

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

        # Display the prediction
        if str(pred_class) == 'mtp':
            st.success("This is Son Tung MTP from Vietnam!")
        else:
            st.success("This is G-Dragon from Korea.")

        # Display the image
        display_img = mpimg.imread(file_path)
        st.image(display_img, use_column_width=True)

    else:
        pass

if __name__ == "__main__":
    main()