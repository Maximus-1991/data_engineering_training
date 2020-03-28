import numpy as np
import os

from config import *
from src.utils.dump_load_pickle import load_data_batch, pickle_data

def generate_train_test_val_dataset():
    dataset_dir = os.path.join(DATA_DIR, "raw", "cifar-10-batches-py")

    # TODO: retrieve number of samples from dataset_dir file
    X_train = np.zeros((40000, 32, 32, 3), dtype="float32")
    y_train = np.zeros((40000, 1), dtype="ubyte").flatten()
    n_samples = 10000  # aantal samples per batch

    for i in range(0, 4):
        batch_name = os.path.join(dataset_dir, "data_batch_" + str(i + 1))
        batch = load_data_batch(batch_name)
        X_train[i * n_samples:(i + 1) * n_samples] = (batch['data'].reshape(-1, 32, 32, 3) / 255.).astype(
            "float32")
        y_train[i * n_samples:(i + 1) * n_samples] = np.array(batch['labels'], dtype='ubyte')

    # validation set, batch 5
    val_batch = load_data_batch(os.path.join(dataset_dir, "data_batch_5"))
    X_val = (val_batch['data'].reshape(-1, 32, 32, 3) / 255.).astype("float32")
    y_val = np.array(val_batch['labels'], dtype='ubyte')

    # test set
    test_batch = load_data_batch(os.path.join(dataset_dir, "test_batch"))
    X_test = (test_batch['data'].reshape(-1, 32, 32, 3) / 255.).astype("float32")
    y_test = np.array(test_batch['labels'], dtype='ubyte')

    # labels
    meta_dict = load_data_batch(os.path.join(dataset_dir, "batches.meta"))
    label_to_names = {k: v for k, v in zip(range(10), meta_dict['label_names'])}

    dataset_dict = {'X_train':X_train, 'y_train':y_train, 'X_val':X_val, 'y_val':y_val, 'X_test':X_test, 'y_test':y_test, 'label_to_names': label_to_names}
    pickle_dict = {os.path.join(DATASET_DIR, f"{k}.pickle"):v for k,v in dataset_dict.items()}
    for k,v in pickle_dict.items():
        pickle_data(data=v, pickle_file_path=k)

    print("training set size: data = {}, labels = {}".format(X_train.shape, y_train.shape))
    print("validation set size: data = {}, labels = {}".format(X_val.shape, y_val.shape))
    print("Test set size: data = " + str(X_test.shape) + ", labels = " + str(y_test.shape))

    return X_train, y_train, X_val, y_val, X_test, y_test, label_to_names