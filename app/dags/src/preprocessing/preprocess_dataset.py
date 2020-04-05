import numpy as np
import pickle
import logging
import os

from src.config import *
from src.utils.dump_load_pickle import load_pickle

def calc_mean_std():
    X_train = load_pickle(os.path.join(DATASET_DIR, "X_train.pickle"))
    # Conv nets trainen duurt erg lang op CPU, dus we gebruiken maar een klein deel
    # van de data nu, als er tijd over is kan je proberen je netwerk op de volledige set te runnen
    X_train[:10000]
    mean = np.mean(X_train)
    std = np.std(X_train)

    with open(MEAN_STD_PICKLE_PATH, 'wb') as f:
        pickle.dump([mean, std], f)

def normalize(data):
    with open(MEAN_STD_PICKLE_PATH, 'rb') as f:
        mean, std = pickle.load(f)
    return (data - mean) / std

def preprocess_data(data):
    if not os.path.exists(MEAN_STD_PICKLE_PATH):
        calc_mean_std()

    preprocessed_data = normalize(data)
    logging.info("Preprocessed data")
    return preprocessed_data


