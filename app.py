from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

model = load_model('model.h5')

def face_recognition(image):
    processed_image = preprocess_image(image)

    predictions = model.predict(np.expand_dims(processed_image, axis=0))

    return predictions

def preprocess_image(image):
    # resized_image = image.resize((width, height))
    numpy_image = np.array(image)
    preprocessed_image = numpy_image / 255.0

    return preprocessed_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    uploaded_image = Image.open(file)

    predictions = face_recognition(uploaded_image)

    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)