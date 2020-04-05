import numpy as np
import os
from tensorflow.keras.models import load_model

from src.config import *
from src.preprocessing.preprocess_dataset import preprocess_data

def predict(X):
    preprocess_data(X)
    assert os.path.exists(MODEL_PATH), "Model does not exist"
    model = load_model(MODEL_PATH, custom_objects=None, compile=True)
    predictions = np.array(model.predict(X, batch_size=100))
    # Take the highest prediction
    predictions = np.argmax(predictions, axis=1)

    return predictions


