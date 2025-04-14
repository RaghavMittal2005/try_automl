import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the trained CNN model
model = load_model('cnn_model.h5')

st.title("CIFAR-10 Image Classifier with CNN")
# Class names for CIFAR-10
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Decode and resize image to 32x32 (RGB)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32, 32))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize and reshape for CNN
    img_rgb = img_rgb.astype('float32') / 255.0
    img_rgb = np.expand_dims(img_rgb, axis=0)  # Add batch dimension (1, 32, 32, 3)

    # Predict using the CNN model
    prediction = model.predict(img_rgb)

    predicted_class = class_names[np.argmax(prediction)]
    st.image(img_rgb[0], caption='Uploaded Image (Resized to 32x32)',  use_container_width=True)
    st.write("Prediction:", predicted_class)
