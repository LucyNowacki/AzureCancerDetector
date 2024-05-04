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

app = Flask(__name__)

# Load the model and scaler at app startup
# Assuming your `rf_model.joblib` and `fitted_scaler.joblib` are in the root directory of your Azure deployment
model_path = os.path.join(os.getcwd(), 'rf_model.joblib')
model = load(model_path)
scaler_path = os.path.join(os.getcwd(), 'fitted_scaler.joblib')
scaler = load(scaler_path)


@app.route('/')
def index():
    # Display the form for user input
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_index = int(request.form['image_index'])
        X = np.load('./Data/X.npy')  # Load data
        
        print("Selected image shape:", X[image_index].shape)  # Should print (50, 50, 3)

        # Flatten the image for prediction
        image_flat = X[image_index].reshape(-1)  # Flatten the image
        print("Flattened image shape:", image_flat.shape)  # Should print (7500,)

        # Transform the data using the pre-fitted scaler
        image_scaled = scaler.transform([image_flat])  # Make sure it's a 2D array for scaling
        
        # Predict using the loaded model
        prediction = model.predict(image_scaled)
        result = 'No Cancer' if prediction[0] == 0 else 'Cancer'
        
        # Convert image to Base64 for displaying
        fig, ax = plt.subplots()
        ax.imshow(X[image_index])  # Use original shape for displaying
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
    app.run(debug=True, host='0.0.0.0', port=8000)







