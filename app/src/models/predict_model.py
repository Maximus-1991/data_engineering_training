import numpy as np
import os
from tensorflow.keras.models import load_model

from src.config import *
from src.preprocessing.preprocess_dataset import preprocess_data

def predict(X):
    preprocess_data(X)
    assert os.path.exists(MODEL_PATH), "Model does not exist"
    model = load_model(MODEL_PATH, custom_objects=None, compile=True)
    class_probs = np.array(model.predict(X, batch_size=100))
    # Take the highest prediction
    prediction = np.argmax(class_probs, axis=1)
    #TODO: add prediction idx (np.argmax) but also decoded label (e.g. dog)
    return class_probs, prediction


