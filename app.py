import cv2
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
import streamlit as st
from fpdf import FPDF

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

# Define a PDF generator class
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Diabetic Retinopathy Detection', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page %s' % self.page_no(), 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, text):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, text)
        self.ln(10)
 

# Initialize Streamlit app
def main():
    st.title("Diabetic Retinopathy Detection")
    
    # Upload image files
    uploaded_files = st.file_uploader("Upload images (Max 2)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        # Process each uploaded image
        images = []
        for uploaded_file in uploaded_files:
            # Save the uploaded file temporarily
            image_path = 'uploaded_image.jpg'
            with open(image_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            # Load the original uploaded image
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Preprocess the uploaded image
            preprocessed_image = load_ben_color(image_path)
            
            # Reshape the image for model input
            input_image = np.expand_dims(preprocessed_image, axis=0)
            
            # Make prediction
            model = load_model()
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
        
        # Clear images if more than 2 are uploaded
        if len(images) > 2:
            images = images[-2:]
        
        # Display the images
        if len(images) > 0:
            st.subheader("Uploaded Images")
            for i, (original_image, preprocessed_image, class_name) in enumerate(images):
                st.subheader(f"Image {i+1}")
                st.image(original_image, use_column_width=True, caption="Original Image")
                st.image(preprocessed_image, use_column_width=True, caption="Preprocessed Image")
                st.write(f"Prediction: {class_name}")
                st.write("---")
        
   # Print button
    if st.button("Print"):
        st.write("Printing...")
        # Generate the PDF document
        pdf = PDF()
        pdf.add_page()
        
        # Add the images and predictions to the PDF
        for i, (original_image, preprocessed_image, class_name) in enumerate(images):
            pdf.chapter_title(f"Image {i+1}")
            pdf.image(original_image, w=150)
            pdf.image(preprocessed_image, w=150)
            pdf.chapter_body(f"Prediction: {class_name}")
        
        # Save the PDF file
        pdf.output("diabetic_retinopathy_report.pdf", "F")
        
        # Provide a download link for the PDF file
        st.download_button("Download PDF", "diabetic_retinopathy_report.pdf")
        
if __name__ == '__main__':
    main()
