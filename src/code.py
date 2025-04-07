import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from PIL import Image

# Load and prepare the MNIST dataset (outside the app's main function)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Model architecture
network = Sequential([
    Dense(512, activation='relu', input_shape=(28 * 28,)),
    Dense(10, activation='softmax')
])

# Compile the model
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare the image data
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

# Prepare the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Train the model (outside the app)
history = network.fit(train_images, train_labels, epochs=5, batch_size=128, validation_data=(test_images, test_labels))

# Function to preprocess the input image
def process_image(image):
    img = image.resize((28, 28)).convert('L')  # Resize and convert to grayscale
    img = np.array(img)
    img = img.astype('float32') / 255  # Normalize
    return img.reshape((1, 784))  # Reshape for prediction

# Streamlit app
def main():
    st.title("MNIST Digit Recognition with Streamlit")

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        .sidebar .sidebar-content .block-container {
            margin: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar
    st.sidebar.title("Model Info")
    st.sidebar.info("This app recognizes handwritten digits using a neural network model trained on the MNIST dataset.")

    # Model summary
    st.sidebar.subheader("Model Architecture")
    st.sidebar.text(network.summary())

    # Training accuracy and loss
    st.sidebar.subheader("Training Metrics")
    st.sidebar.text(f"Final Accuracy: {history.history['accuracy'][-1]:.4f}")
    st.sidebar.text(f"Final Loss: {history.history['loss'][-1]:.4f}")

    # Main content area
    st.markdown("---")
    st.header("Digit Recognition")

    # Upload image
    uploaded_file = st.file_uploader("Choose a handwritten digit image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img.resize((100, 100)), caption="Uploaded Image", use_column_width=False)

        img_array = process_image(img)  # Preprocess image
        prediction = network.predict(img_array)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.subheader("Prediction")
        st.write(f"Predicted digit: {predicted_digit}")
        st.write(f"Confidence: {confidence:.2f}%")

    # Button to toggle between prediction and training metrics
    if st.button("Toggle Training Metrics"):
        st.subheader("Training Metrics Over Epochs")
        st.line_chart(history.history['accuracy'], use_container_width=True)
        st.line_chart(history.history['loss'], use_container_width=True)

if __name__ == "__main__":
    main()
