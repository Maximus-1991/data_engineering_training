import numpy as np
import os
from sklearn.metrics import classification_report

from config import *
from src.utils.dump_load_pickle import load_pickle

def evaluate_model(predictions, y):
    y = load_pickle(os.path.join(DATASET_DIR, f"{y}.pickle"))
    y = np.array(y, dtype=np.int32)
    label_to_names = load_pickle(os.path.join(DATASET_DIR, "label_to_names.pickle"))
    # Print model performance results
    print("Accuracy = {}".format(np.sum(predictions == y) / float(len(predictions))))
    print(classification_report(y, predictions, target_names=list(label_to_names.values())))
