import os

#PATHS
#PROJECT_ROOT = "/home/maartens/Documents/Programming/Projects/data_engineering_training/app"
PROJECT_ROOT = "/usr/local/airflow"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "interim")
PROCESSED_DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

#MODEL PATHS
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, 'model.h5')
MEAN_STD_PICKLE_PATH = os.path.join(MODEL_DIR, "mean_std.pickle")

#MODEL_PARAMETERS
MODEL_INPUT_SHAPE = (32, 32, 3)
#NR_CHANNELS = MODEL_INPUT_SHAPE[2]
#IMAGE_SIZE = MODEL_INPUT_SHAPE[0]
NR_CLASSES = 10
EPOCHS = 20
BATCH_SIZE = 50
