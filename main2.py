import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/plant_disease_prediction_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices1 = json.load(open(f"{working_dir}/class_indices.json", encoding='utf-8'))



#Sidebar 
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Leaves Recognition"])

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices1):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices1[str(predicted_class_index)]
    return predicted_class_name

#Main Page
if(app_mode=="Home"):
    st.header("PLANT LEAF RECOGNITION SYSTEM")
    image_path = "pexels-photo.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Leaf Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant leaf efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of leaves. Together, let's identify the medical leaves of plants!

    ### How It Works
    1. **Upload Image:** Go to the **Leaves Recognition** page and upload an image of a plant Leaf.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify Medical Plant Name.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate Medical Plant Name detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Plant Leaf Recognition** page in the sidebar to upload an image and experience the power of our Plant Leaf Name Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                  #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                A new directory containing 80 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (80 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Leaves Recognition"):
    st.title('PLANT LEAF RECOGNITION SYSTEM')
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((450, 450))
            st.image(resized_img)

        with col2:
            if st.button('Prediction'):
            # Preprocess the uploaded image and predict the class
                prediction = predict_image_class(model, uploaded_image, class_indices1)
                st.success(f'Prediction: {str(prediction)}')
