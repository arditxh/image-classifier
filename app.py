import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('cnn_flower_classification_model.h5')

# Define the class names
class_names = ["Lily", "Lotus", "Orchid", "Sunflower", "Tulip"]

def load_and_prepare_image(img_path, img_size=(150, 150)):
    """
    Load an image and prepare it for prediction by resizing and scaling.

    Args:
    img_path (str): Path to the image file.
    img_size (tuple): Target size for the image (width, height).

    Returns:
    np.array: Preprocessed image ready for model prediction.
    """
    # Load the image
    img = image.load_img(img_path, target_size=img_size)
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Scale the image
    img_array = img_array / 255.0
    
    # Expand dimensions to match the shape required by the model
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, img

# Set the background color and style
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .title {
        font-size: 42px;
        color: #4B9CD3;
        text-align: center;
        font-weight: bold;
    }
    .subtitle {
        font-size: 24px;
        color: #333333;
        text-align: center;
        font-weight: normal;
    }
    .footer {
        font-size: 16px;
        color: #888888;
        text-align: center;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and subtitle
st.markdown('<div class="title">Flower Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image to classify it as Lily, Lotus, Orchid, Sunflower, or Tulip</div>', unsafe_allow_html=True)

# Upload an image
uploaded_file = st.file_uploader("", type="jpg")

if uploaded_file is not None:
    # Save the uploaded file temporarily to disk
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Preprocess the image
    img_array, img = load_and_prepare_image("temp_image.jpg", img_size=(150, 150))

    # Display the image
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]

    # Display the prediction result
    st.markdown(f"<h2 style='text-align: center; color: #4B9CD3;'>Prediction: {predicted_class_name}</h2>", unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Developed with ❤️ using Streamlit</div>', unsafe_allow_html=True)
