import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template, jsonify
from keras.src.applications.densenet import DenseNet121
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.applications import DenseNet121
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

try:
    base_model = DenseNet121(include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    flat = Flatten()(x)
    class_1 = Dense(4608, activation='relu')(flat)
    drop_out = Dropout(0.2)(class_1)
    class_2 = Dense(1152, activation='relu')(drop_out)
    output = Dense(2, activation='softmax')(class_2)
    model_03 = Model(base_model.inputs, output)

    # Load weights
    MODEL_PATH = 'DenseNet121 - Copie.h5'  # Update this if your weights file has a different name
    model_03.load_weights(MODEL_PATH)
    app.logger.info("Model loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading model: {str(e)}")
    raise

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to map class index to class name
def get_class_name(class_no):
    if class_no == 0:
        return "Normal"
    elif class_no == 1:
        return "Pneumonia"
    return "Unknown"

# Function to process image and get prediction directly from stream
def get_result(file_stream):
    try:
        # Read directly from file stream
        image = Image.open(file_stream).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image)
        input_img = np.expand_dims(image, axis=0) / 255.0  # Normalize to [0,1]

        # Make prediction
        result = model_03.predict(input_img)
        class_index = np.argmax(result, axis=1)[0]
        confidence = float(np.max(result))

        return class_index, confidence
    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        raise

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            app.logger.error("No file part in request")
            return jsonify({'error': 'No file uploaded'}), 400

        f = request.files['file']
        if f.filename == '':
            app.logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400

        try:
            # Get prediction directly from stream
            class_index, confidence = get_result(f.stream)
            result = get_class_name(class_index)

            # Return JSON response
            response = {
                'prediction': result,
                'confidence': f"{confidence:.2%}"
            }
            app.logger.debug(f"Prediction successful: {response}")
            return jsonify(response)

        except Exception as e:
            app.logger.error(f"Prediction error: {str(e)}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Method not allowed'}), 405

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)