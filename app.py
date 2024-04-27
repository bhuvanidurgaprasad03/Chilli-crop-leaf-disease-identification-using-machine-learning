# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model

# Load the pre-trained model (Assuming you can't modify the model)
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
try:
    # Attempt to load the model with the original configuration
    model = model_from_json(loaded_model_json)
except ValueError as e:
    if "Unrecognized keyword arguments" in str(e):
        # Handle the case where 'batch_shape' argument is present
        print("WARNING: Your model might have been created with an unsupported 'batch_shape' argument. Using CPU for prediction.")
        from tensorflow.keras.layers import Input
        # Define a new Input layer without 'batch_shape'
        input_layer = Input(shape=(100, 100, 3))
        # Rest of your model definition here... (assuming you have the remaining layers)
        model = Model(inputs=input_layer, outputs=None)  # Build the model

model.load_weights("model_weights.weights.h5")

# Define your categories
CATEGORIES = ['Bacterial leaf spot', 'Healthy', 'Leaf Spot', 'Mosaicvirus', 'Powdery mildew']

# Add information about diseases
disease_info = {
    'Bacterial leaf spot': 'Bacterial leaf spot is a common disease affecting chili plants. It is caused by bacteria and can lead to significant yield loss if not controlled.',
    'Healthy': 'Your plant appears to be healthy. Keep up with proper care and maintenance to ensure continued health.',
    'Leaf Spot': 'Leaf spot is a fungal disease that affects chili plants. It can cause yellowing and spotting on leaves, leading to reduced plant vigor.',
    'Mosaicvirus': 'Mosaic virus is a viral disease commonly found in chili plants. It causes mottling and discoloration of leaves, affecting plant growth and yield.',
    'Powdery mildew': 'Powdery mildew is a fungal disease that affects many plants, including chili. It appears as a white powdery coating on leaves and can stunt plant growth.'
}

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    # Read and preprocess the image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (100, 100))
    img = img / 255.0
    
    # Predict disease
    prediction = model.predict(np.array([img]))
    predicted_class = np.argmax(prediction)
    result = CATEGORIES[predicted_class]
    
    # Get additional information about the predicted disease
    disease_additional_info = disease_info.get(result, "Additional information not available.")
    
    # Prepare JSON response
    response = {
        "result": result,
        "info": disease_additional_info
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
