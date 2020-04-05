import numpy as np
from flask import Flask, request

from src.models.predict_model import predict

app = Flask(__name__)

@app.route("/")
def hello():
    return "<h1 style='color:blue'>Hello There!</h1>"

@app.route('/predict', methods=['POST'])
def predict_json():
    content = request.json
    image = np.array(content)
    image = np.moveaxis(image, 0, -1) # make color channel the 3rd dimension
    image = np.array([image]) #input of model is (batch_size, 32, 32, 3)
    prediction = predict(image).tolist()
    print(prediction)
    return {'prediction':prediction}


if __name__ == "__main__":
    app.run(host='0.0.0.0')