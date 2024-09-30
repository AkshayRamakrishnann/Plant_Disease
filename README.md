# Plant Disease Prediction App

## Overview

The Plant Disease Prediction App is a web application built using Streamlit that allows users to upload images of plant leaves to identify potential diseases. This app utilizes deep learning models to analyze the uploaded images and provide predictions based on the visual characteristics of the leaves. The app also incorporates the OpenVINO library to optimize the model for inference, improving performance and enabling deployment on various hardware platforms.

## Features

- **Image Upload**: Users can upload images of plant leaves.
- **Disease Prediction**: The app provides predictions of potential diseases based on the uploaded images.
- **Optimized Inference**: The app leverages the OpenVINO toolkit to enhance model inference speed and efficiency.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Optimization with OpenVINO](#model-optimization-with-openvino)
- [License](#license)

## Installation

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

## Install Dependencies
###Create a virtual environment (optional but recommended):

### Clone the Repository
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

### Install the required libraries:
`pip install -r requirements.txt`

## How to Use the App

1. **Upload an Image**: Click on the "Upload Image" button to select an image of a plant leaf.
2. **View Prediction**: After the image is uploaded, the app will display the predicted disease based on the model’s analysis.
3. **Chat with the Chatbot**: You can also interact with a simple chatbot to ask questions about the predictions.

## Model Optimization with OpenVINO

OpenVINO (Open Visual Inference and Neural Network Optimization) is an Intel toolkit that optimizes deep learning models for inference. In this project, we effectively utilize OpenVINO to enhance the performance of our plant disease classification model. Here are the key aspects of how OpenVINO is used:

### Model Conversion

- **Model Optimization**: The original model trained with TensorFlow is converted into an Intermediate Representation (IR) format using the Model Optimizer provided by OpenVINO. This process involves:
  - **Quantization**: Reducing the precision of the model's weights to lower bit-width, which decreases the model size and improves inference speed.
  - **Layer Fusion**: Merging layers in the model graph to reduce computational overhead.

- **Command to Convert Model**: The model can be converted using the following command:
  ```bash
  mo --input_model model.h5 --output_dir output_directory

### Inference Optimization

Once the model is converted to the IR format, it can be loaded and executed efficiently using OpenVINO's Inference Engine. This allows for:

- **Cross-Platform Deployment**: The optimized model can run on various Intel hardware, including CPUs, GPUs, and VPUs, without needing to change the code.
- **Faster Inference Times**: By leveraging Intel’s optimizations, the app provides real-time predictions, making it suitable for deployment in agricultural settings.

- The trained model can be found here https://drive.google.com/drive/folders/1F6Uo83biNN0gGOMZCbKAtGoQsn9S7Ypx?usp=drive_link

### Performance Evaluation

After integrating OpenVINO, the app showed significant improvements in inference times compared to the original TensorFlow model, allowing it to handle multiple requests simultaneously and providing a better user experience.

```bash
git clone https://github.com/yourusername/plant-disease-prediction-app.git
cd plant-disease-prediction-app



