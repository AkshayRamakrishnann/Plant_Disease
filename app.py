import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the pre-trained DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Load the plant disease classification model
model_plant = tf.keras.models.load_model('plant_disease_model.h5')

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

# Define the image prediction function
def predict_image(model, img):
    img = img.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    if predicted_class_index < len(class_labels):
        predicted_class_label = class_labels[predicted_class_index]
    else:
        predicted_class_label = "Unknown class"  # Default in case of error

    return predicted_class_label

# Chat with the DialoGPT chatbot
def chatbot_response(user_input, chat_history_ids=None):
    # Tokenize the user input and concatenate it with the chat history (if any)
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    # Generate response using the model
    bot_output = model.generate(torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids,
                                max_length=1000, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the bot's response
    response = tokenizer.decode(bot_output[:, new_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return response, bot_output

# Initialize chat history
chat_history = None

# Streamlit UI
st.title("Plant Disease Prediction and Chatbot")
st.write("Upload an image of a plant leaf to get a disease prediction or chat with the AI-powered chatbot.")

# File uploader for plant disease classification
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    if st.button("Classify"):
        prediction = predict_image(model_plant, img)
        st.write(f"Predicted Class: {prediction}")

# Chatbot interface
st.write("## Chat with the AI-powered Chatbot")
user_input = st.text_input("Ask a question to the chatbot:")

if user_input:
    response, chat_history = chatbot_response(user_input, chat_history)
    st.write(f"Chatbot: {response}")
