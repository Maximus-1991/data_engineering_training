import numpy as np
from tensorflow.keras.models import load_model

from config import *
from src.preprocessing.preprocess_dataset import *
from src.utils.dump_load_pickle import load_pickle


def predict(X):
    X_pickle_path = os.path.join(PROCESSED_DATASET_DIR, f"{X}.pickle")
    if not os.path.exists(X_pickle_path):
        preprocess_data(X)

    X = load_pickle(X_pickle_path)
    assert os.path.exists(MODEL_PATH), "Model does not exist"
    model = load_model(MODEL_PATH, custom_objects=None, compile=True)
    predictions = np.array(model.predict(X, batch_size=100))
    # Take the highest prediction
    predictions = np.argmax(predictions, axis=1)

    return predictions



