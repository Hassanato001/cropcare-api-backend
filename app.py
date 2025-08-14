import os
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS
import tensorflow as tf
from PIL import Image
import numpy as np

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask app
app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing your frontend to connect

# --- Load the Trained Model ---
# This path assumes your model file is in the root directory of your server
MODEL_PATH = 'crop_disease_model.keras' 
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --- IMPORTANT: Verify Class Names ---
# The order MUST match the order found during training.
# Check your final Colab output for "Found the following classes: [...]"
CLASS_NAMES = ['cbsd', 'cmd', 'healthy']

# --- Define the Prediction Function ---
def process_image(image_file):
    """
    Processes the uploaded image to the format the model expects.
    """
    img = Image.open(image_file.stream).convert('RGB') # Ensure image is in RGB
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    return img_array

# --- Define the API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded, check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    if file:
        try:
            # 1. Process the image
            processed_image = process_image(file)

            # 2. Make a prediction
            prediction_scores = model.predict(processed_image)[0]
            
            # 3. Apply the confidence threshold logic
            CONFIDENCE_THRESHOLD = 0.75  # 75%
            confidence_score = float(np.max(prediction_scores))

            if confidence_score < CONFIDENCE_THRESHOLD:
                # If confidence is too low, return an "Uncertain" response
                response = {
                    'prediction': {
                        'class_name': 'Uncertain',
                        'confidence': f"{confidence_score:.2%}",
                        'error': 'Could not confidently identify a cassava leaf. Please use a clearer image.'
                    }
                }
            else:
                # If confidence is high enough, return the class name
                predicted_class_index = np.argmax(prediction_scores)
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                response = {
                    'prediction': {
                        'class_name': predicted_class_name,
                        'confidence': f"{confidence_score:.2%}"
                    }
                }
            
            return jsonify(response)
        
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

    return jsonify({'error': 'An unknown error occurred'}), 500

# Health check route
@app.route('/', methods=['GET'])
def index():
    return "CropCare AI Backend is running."

# This block is not needed for Hugging Face Spaces deployment
# if __name__ == '__main__':
#     app.run(debug=True)
