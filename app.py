import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import json
import pickle
import logging
import time

from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the model
kmeans_model = pickle.load(open('kmeans_model.pkl', 'rb'))
scalar = pickle.load(open('standard_scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    logger.info(f"Received data for prediction: {data}")
    input_array = np.array([list(item.values()) for item in data]).reshape(len(data), -1)
    logger.info(f"Reshaped input array: {input_array}")

    new_data = scalar.transform(input_array)

    # Measure response time
    start_time = time.time()
    output = kmeans_model.predict(new_data)
    response_time = time.time() - start_time

    logger.info(f"Prediction output: {output} with response time: {response_time:.4f} seconds")

    return jsonify({'predictions': [{'cluster': cluster} for cluster in output], 'response_time': response_time})

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input data from the form
    data = [
        float(request.form['alcohol']),
        float(request.form['malic_acid']),
        float(request.form['ash']),
        float(request.form['alcalinity_of_ash']),
        float(request.form['magnesium']),
        float(request.form['total_phenols']),
        float(request.form['flavanoids']),
        float(request.form['nonflavanoid_phenols']),
        float(request.form['proanthocyanins']),
        float(request.form['color_intensity']),
        float(request.form['hue']),
        float(request.form['od280_od315']),
        float(request.form['proline'])
    ]
    
    logger.info(f"Received form data for prediction: {data}")
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    logger.info(f"Transformed input for prediction: {final_input}")

    # Measure response time
    start_time = time.time()
    output = kmeans_model.predict(final_input)[0]
    response_time = time.time() - start_time

    logger.info(f"Prediction output: {output} with response time: {response_time:.4f} seconds")

    return render_template("home.html", prediction_text="The clustering_id is {}".format(output), response_time=response_time, cluster_id=output)

if __name__ == "__main__":
    app.run(debug=True)
