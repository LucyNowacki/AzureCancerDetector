from flask import Flask, request, jsonify, render_template
from joblib import load
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Configuration settings from environment variables
PORT = int(os.getenv('PORT', 8000))
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False') == 'True'
MODEL_PATH = os.getenv('MODEL_PATH', 'rf_model.joblib')
SCALER_PATH = os.getenv('SCALER_PATH', 'fitted_scaler.joblib')

app = Flask(__name__)

# Load the model and scaler at app startup
model_path = os.path.join(os.getcwd(), MODEL_PATH)
model = load(model_path)

scaler_path = os.path.join(os.getcwd(), SCALER_PATH)
scaler = load(scaler_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_index = int(request.form['image_index'])
        X = np.load('./Data/X.npy')  # Load data
        print("Selected image shape:", X[image_index].shape)
        image_flat = X[image_index].reshape(-1)
        print("Flattened image shape:", image_flat.shape)
        image_scaled = scaler.transform([image_flat])
        prediction = model.predict(image_scaled)
        result = 'No Cancer' if prediction[0] == 0 else 'Cancer'
        fig, ax = plt.subplots()
        ax.imshow(X[image_index])
        ax.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        return jsonify({'prediction': result, 'image': image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__=='__main__':
    app.run(host='0.0.0.0', port=PORT)




