# Pneumonia Detection from Chest X-rays

This project uses a convolutional neural network (CNN) model to detect pneumonia from chest X-ray images. The model is hosted on a Streamlit app where users can upload X-ray images and receive a diagnosis indicating whether pneumonia is likely present.

## Features

- **X-ray Image Upload**: Allows users to upload chest X-ray images in JPG, JPEG, or PNG formats.
- **Pneumonia Detection**: The CNN model predicts whether the uploaded X-ray shows signs of pneumonia.
- **Confidence Score**: Displays the model’s confidence level for each prediction.

## Tech Stack

- **TensorFlow/Keras** for the CNN model
- **Streamlit** for the web interface
- **Pillow** for image processing

## Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/visionjain/pneumonia-detection-X-ray
    cd pneumonia-detection-X-ray
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the pre-trained model**:
   Place your `trained.h5` model file in the root directory.

4. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

## Usage

1. Launch the app, and upload a chest X-ray image.
2. The model will display a prediction (`Pneumonia` or `Normal`) along with a confidence level.
3. Review the results to understand the model's prediction of the image.

