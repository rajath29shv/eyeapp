import cv2
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
import streamlit as st
import streamlit_reports as sfr

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

# Initialize Streamlit app
st.title("Diabetic Retinopathy Detection")

# Define the main app logic
def main():
    uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True)
    
    if uploaded_files:
        for file in uploaded_files:
            # Save the uploaded file temporarily
            image_path = 'uploaded_image.jpg'
            with open(image_path, "wb") as f:
                f.write(file.read())

            # Load the original uploaded image
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            # Preprocess the uploaded image
            preprocessed_image = load_ben_color(image_path)

            # Reshape the image for model input
            input_image = np.expand_dims(preprocessed_image, axis=0)

            # Make prediction
            prediction = predict_image(tf.convert_to_tensor(input_image))
            class_id = np.argmax(prediction)
            class_name = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'][class_id]

            # Convert images to base64 encoding
            original_image_base64 = image_to_base64(original_image)
            preprocessed_image_base64 = image_to_base64(preprocessed_image)

            # Display the images and prediction result
            st.image(original_image, caption="Original Image", use_column_width=True)
            st.image(preprocessed_image, caption="Preprocessed Image", use_column_width=True)
            st.write("Prediction:", class_name)
            st.write("---")
            
            # Add the image and prediction result to the report page
            page = sfr.Page()
            page.image(original_image, caption="Original Image", use_column_width=True)
            page.image(preprocessed_image, caption="Preprocessed Image", use_column_width=True)
            page.write("Prediction:", class_name)

            # Add the page to the list of pages
            pages.append(page)

        # Generate the report PDF
        pdf = sfr.create_report(pages)

        # Convert the PDF to base64 encoding
        pdf_base64 = base64.b64encode(pdf).decode('utf-8')

        # Create a download link for the PDF
        href = f'<a href="data:application/pdf;base64,{pdf_base64}" download="report.pdf">Download Report</a>'
        st.markdown(href, unsafe_allow_html=True)


@tf.function
def predict_image(image):
    return model(image)

def image_to_base64(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', image_rgb)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64

# Run the Streamlit app
if __name__ == '__main__':
    main()
