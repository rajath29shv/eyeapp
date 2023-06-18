import cv2
import tensorflow as tf
import numpy as np
import streamlit as st
from io import BytesIO

# Define the custom FixedDropout layer
class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = tf.keras.backend.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

# Register the custom FixedDropout layer with Keras
tf.keras.utils.get_custom_objects()['FixedDropout'] = FixedDropout

input_shape = (224, 224, 3)

def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (input_shape[0], input_shape[1]))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image

# Define the image cropping function (you can customize this based on your needs)
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

# Load the trained model
model = tf.keras.models.load_model('diabetic_retinopathy_detection_model.h5')

def predict_image(image):
    return model(image)

@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('diabetic_retinopathy_detection_model.h5')

# Initialize Streamlit app
def main():
    st.title("Diabetic Retinopathy Detection")
    
    # Upload image files
    uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        # Process each uploaded image
        images = []
        for uploaded_file in uploaded_files:
            # Read image file as bytes
            image_bytes = uploaded_file.read()

            # Load the original uploaded image
            original_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            # Preprocess the uploaded image
            preprocessed_image = load_ben_color(BytesIO(image_bytes))

            # Reshape the image for model input
            input_image = np.expand_dims(preprocessed_image, axis=0)

            # Load the model
            model = load_model()

            # Make prediction
            prediction = predict_image(tf.convert_to_tensor(input_image))
            class_id = np.argmax(prediction)
            class_name = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'][class_id]

            # Display the original and preprocessed images
            st.subheader("Original Image")
            st.image(original_image, use_column_width=True)

            st.subheader("Preprocessed Image")
            st.image(preprocessed_image, use_column_width=True)

            # Display the predicted class
            st.subheader("Prediction")
            st.write(f"Class: {class_name}")
            st.write("---")

            images.append((original_image, preprocessed_image, class_name))

        # Display the last two uploaded images
        if len(images) >= 2:
            st.subheader("Uploaded Images")
            for i in range(len(images)-2, len(images)):
                original_image, preprocessed_image, class_name = images[i]
                st.subheader(f"Image {i+1}")
                st.image(original_image, use_column_width=True, caption="Original Image")
                st.image(preprocessed_image, use_column_width=True, caption="Preprocessed Image")
                st.write(f"Prediction: {class_name}")
                st.write("---")
        else:
            st.warning("Please upload at least two images.")
    
    else:
        st.warning("Please upload some images.")

# Run the Streamlit app
if __name__ == '__main__':
    main()
