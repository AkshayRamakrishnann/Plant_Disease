import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

# Load the model
model = tf.keras.models.load_model('plant_disease_model.h5')
print("Model output shape:", model.output_shape)  # Check the model output shape

# Define class labels
class_labels = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
    'Blueberry__healthy', 'Cherry_Powdery_mildew', 'Cherry_healthy', 'Corn__Cercospora_leaf_spot Gray_leaf_spot',
    'Corn__Common_rust', 'Corn_Northern_Leaf_Blight', 'Corn_healthy', 'Grape__Black_rot',
    'Grape__Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange__Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach_healthy', 'Pepper,_bell__Bacterial_spot',
    'Pepper,bell_healthy', 'Potato_Early_blight', 'Potato_Late_blight', 'Potato__healthy',
    'Raspberry__healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry__Leaf_scorch',
    'Strawberry__healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato__Late_blight',
    'Tomato__Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato__healthy'
]

print("Number of class labels:", len(class_labels))  # Check the number of class labels

# Define the image prediction function
def predict_image(model, img):
    img = img.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    print("Predictions:", predictions)  # Print raw predictions

    predicted_class_index = np.argmax(predictions, axis=1)[0]
    print("Predicted class index:", predicted_class_index)  # Print predicted index

    if predicted_class_index < len(class_labels):
        predicted_class_label = class_labels[predicted_class_index]
    else:
        predicted_class_label = "Unknown class"  # Default in case of error

    return predicted_class_label

# Streamlit UI
st.title("Plant Disease Prediction")
st.write("Upload an image of a plant leaf to get a disease prediction.")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    if st.button("Classify"):
        prediction = predict_image(model, img)
        st.write(f"Predicted Class: {prediction}")
