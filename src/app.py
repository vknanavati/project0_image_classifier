# app.py — the Flask API that exposes our model to the outside world
# This file's only job is to handle HTTP requests and return responses
# It knows nothing about how the model works — it just calls predict.py

import sys
import os
import tempfile
from flask import Flask, request, jsonify

sys.path.append(os.path.dirname(__file__))
import config
from predict import load_model
from predict import predict as predict_image

# Create the Flask app
app = Flask(__name__)

# Load the model once when the server starts up
# Analogy: a doctor prepares their tools before the clinic opens
# not when each patient walks in
print("Loading model...")
model, device = load_model()
print("Model ready. Server starting...")


@app.route('/health', methods=['GET'])
def health():
    # A simple health check endpoint
    # Useful for checking if the server is running
    # Try it in your browser: http://localhost:5000/health
    return jsonify({
        'status': 'ok',
        'model': 'SimpleCNN',
        'classes': config.CLASSES
    })


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    # Main prediction endpoint
    # Accepts an image file upload and returns a prediction
    # 
    # How to call it:
    # curl -X POST -F "image=@yourphoto.jpg" http://localhost:5000/predict

    # Check that an image was actually sent
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided. Send a file with key "image"'}), 400

    file = request.files['image']

    # Check that the file has a name (wasn't empty)
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Check the file extension is an image format we can handle
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    file_ext = file.filename.rsplit('.', 1)[-1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({'error': f'File type not supported. Use: {allowed_extensions}'}), 400

    try:
        # Save the uploaded file to a temporary location on disk
        # We need a file path because our preprocess_image() function
        # expects a path, not raw bytes
        # tempfile.NamedTemporaryFile creates a temp file and auto-deletes it after
        with tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # Run prediction
        result = predict_image(tmp_path, model, device)

        # Clean up the temporary file
        os.unlink(tmp_path)

        # Return the prediction as JSON
        return jsonify({
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence'],
            'confidence_percent': f"{result['confidence']*100:.1f}%",
            'all_probabilities': result['all_probabilities']
        })

    except Exception as e:
        # If anything goes wrong, return a clean error message
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=False  # never run debug=True in production — it exposes internals
    )
