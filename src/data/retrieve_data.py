import os
import shutil
from loguru import logger
import tarfile
from urllib.request import urlretrieve

from config import *

def load_data(data_url):
    dataset_dir = os.path.join(DATA_DIR, "raw")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    data_dir_name = os.path.basename(data_url)
    zip_path = os.path.join(dataset_dir, data_dir_name)
    if not os.path.exists(os.path.join(dataset_dir, data_dir_name.split('.')[0])):
        logger.info("Downloading data...")
        urlretrieve(data_url,
                    os.path.join(dataset_dir, data_dir_name))
        tar = tarfile.open(os.path.join(dataset_dir, data_dir_name))
        tar.extractall(dataset_dir)
        tar.close()

    logger.info("Downloaded data, removing zip file")
    try:
        #shutil.rmtree(zip_path)
        os.remove(zip_path)
    except OSError as e:
        logger.error("Error: %s : %s" % (zip_path, e.strerror))


