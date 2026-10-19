
import PIL
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#Checking Version

print("TensorFlow:", tf.__version__)
print("Streamlit:", st.__version__)
print("NumPy:", np.__version__)
print("Pillow:", PIL.__version__)

# Page configuration
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱",
    layout="centered"
)

# App title
st.title("🐱 Cat vs Dog Classifier")
st.write(
    "Upload an image and let the CNN model predict "
    "whether it contains a cat or a dog."
)

# Load trained model only once
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(
        "cat_dog_model.keras",
        compile=False
    )

model = load_my_model()

# Upload image
uploaded_file = st.file_uploader(
    "Choose a cat or dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image and convert to RGB
    img = Image.open(uploaded_file).convert("RGB")

    # Display uploaded image
    st.image(img, caption="Uploaded Image", width=400)

    # Predict button
    if st.button("Predict Image", type="primary"):

        with st.spinner("Analyzing image..."):

            # Preprocess image
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = float(
                model.predict(img_array, verbose=0)[0][0]
            )

            # Your model uses cats = 0 and dogs = 1
            if prediction >= 0.5:
                label = "Dog 🐶"
                score = prediction
            else:
                label = "Cat 🐱"
                score = 1 - prediction

        # Display result
        st.divider()
        st.subheader("Prediction Result")
        st.success(f"Predicted Class: {label}")
        st.metric("Model Score", f"{score * 100:.2f}%")

        st.caption(
            "The score reflects the model's output for the "
            "predicted class; it is not guaranteed to be "
            "a calibrated probability."
        )

else:
    st.info("Upload an image to get started.")
