import os

from src.config import *
from src.models.create_model import conv_net
from src.preprocessing.preprocess_dataset import preprocess_data
from src.utils.dump_load_pickle import load_pickle


def train_model():
    #Load pickled data
    # Conv nets trainen duurt erg lang op CPU, dus we gebruiken maar een klein deel
    # van de data nu, als er tijd over is kan je proberen je netwerk op de volledige set te runnen
    X_train = load_pickle(os.path.join(DATASET_DIR, "X_train.pickle"))
    X_train = X_train[:10000]
    X_val = load_pickle(os.path.join(DATASET_DIR, "X_val.pickle"))
    y_train = load_pickle(os.path.join(DATASET_DIR, "y_train.pickle"))
    y_train = y_train[:10000]
    y_val = load_pickle(os.path.join(DATASET_DIR, "y_val.pickle"))

    #Preprocess data
    X_train = preprocess_data(X_train)
    X_val = preprocess_data(X_val)

    model = conv_net()
    print(model.summary())
    model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val), verbose=2)
    model.save(MODEL_PATH)
