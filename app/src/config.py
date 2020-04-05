import os

#PATHS
PROJECT_ROOT = "/usr/local"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "interim")
PROCESSED_DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

#MODEL PATHS
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, 'model.h5')
MEAN_STD_PICKLE_PATH = os.path.join(MODEL_DIR, "mean_std.pickle")

#MODEL_PARAMETERS
MODEL_INPUT_SHAPE = (32, 32, 3)
NR_CLASSES = 10
EPOCHS = 20
BATCH_SIZE = 50
