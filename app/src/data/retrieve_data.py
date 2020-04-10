import os
from loguru import logger
import logging
import tarfile
from urllib.request import urlretrieve

from src.config import *

def load_data(data_url):
    dataset_dir = os.path.join(DATA_DIR, "raw")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    data_dir_name = os.path.basename(data_url)
    zip_path = os.path.join(dataset_dir, data_dir_name)
    logging.info('Downloading data...')
    urlretrieve(data_url, os.path.join(dataset_dir, data_dir_name))
    tar = tarfile.open(os.path.join(dataset_dir, data_dir_name))
    tar.extractall(dataset_dir)
    tar.close()

    logging.info('Downloaded data, removing zip file')
    try:
        os.remove(zip_path)
    except OSError as e:
        logger.error("Error: %s : %s" % (zip_path, e.strerror))



