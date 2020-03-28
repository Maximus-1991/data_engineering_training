import os

from config import *
from src.models.create_model import conv_net
from src.preprocessing.preprocess_dataset import *
from src.utils.dump_load_pickle import load_pickle


def train_model():
    input_data_list = ['X_train', 'X_val']

    for input_data in input_data_list:
        if not os.path.exists(os.path.join(PROCESSED_DATASET_DIR, f"{input_data}.pickle")):
            # Create preprocessed data and save pickle
            preprocess_data(input_data)

    #Load preprocessed pickled data
    X_train = load_pickle(os.path.join(PROCESSED_DATASET_DIR, "X_train.pickle"))
    X_val = load_pickle(os.path.join(PROCESSED_DATASET_DIR, "X_val.pickle"))

    #Load y (unprocessed)
    y_train = load_pickle(os.path.join(DATASET_DIR, "y_train.pickle"))
    # Conv nets trainen duurt erg lang op CPU, dus we gebruiken maar een klein deel
    # van de data nu, als er tijd over is kan je proberen je netwerk op de volledige set te runnen
    y_train = y_train[:10000]
    y_val = load_pickle(os.path.join(DATASET_DIR, "y_val.pickle"))

    model = conv_net()
    print(model.summary())
    model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, y_val), verbose=2)
    model.save(MODEL_PATH)
