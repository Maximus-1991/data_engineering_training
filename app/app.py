import numpy as np
from flask import Flask, request

from src.models.predict_model import predict
from src.utils.dump_load_pickle import load_labels

labels = load_labels()

app = Flask(__name__)

@app.route("/")
def hello():
    return "<h1 style='color:blue'>Hello There!</h1>"

@app.route('/predict', methods=['POST'])
def predict_json():
    content = request.json
    image = np.array(content)
    image = np.moveaxis(image, 0, -1)  # make color channel the 3rd dimension
    image = np.array([image]) #input of model is (batch_size, 32, 32, 3)
    class_probs, prediction_enc = predict(image)
    class_probs = class_probs.tolist()
    prediction_enc = int(prediction_enc)
    prediction_dec = labels[prediction_enc]
    prediction_dec = [prediction_dec]
    prediction_enc = [prediction_enc]

    return {'class_probabilities': class_probs, 'prediction_enc': prediction_enc, 'prediction_dec': prediction_dec}

#TODO: Add function to select files on drive and make post request to selected url


if __name__ == "__main__":
    app.run(host='0.0.0.0')