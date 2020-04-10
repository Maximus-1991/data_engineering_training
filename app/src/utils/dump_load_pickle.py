import os
import pickle

from src.config import *

def load_data_batch(batch_name):
    f = open(batch_name, "rb")
    batch = pickle.load(f, encoding="latin1")
    f.close()
    return batch

def pickle_data(data, pickle_file_path):
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        var = pickle.load(f)
    return var

def load_labels():
    labels_dir = os.path.join(DATA_DIR, "raw", "cifar-10-batches-py", 'batches.meta')
    f = open(labels_dir, 'rb')
    cifar_dict = pickle.load(f, encoding='latin1')
    label_to_names = {k: v for k, v in zip(range(10), cifar_dict['label_names'])}
    f.close()
    return label_to_names