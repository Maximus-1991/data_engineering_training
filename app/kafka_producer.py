from time import sleep
from json import dumps, load
from kafka import KafkaProducer
from glob import glob
import os

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x: dumps(x).encode('utf-8'))


def load_json(filename):
    with open(filename, 'r') as f:
        return load(f)


if __name__ == "__main__":
    # Load images
    base_path = os.path.join(os.path.dirname(os.getcwd()), 'app', 'data', 'kafka')
    image_paths = glob(os.path.join(base_path, '*.json'))
    image_names = [os.path.split(path)[-1] for path in image_paths]
    images = [load_json(x) for x in image_paths]
    #Iterate over images
    while True:
        for i, image in enumerate(images):
            producer.send('newImages', key=bytes(image_names[i], encoding='utf-8'), value=image)
            print("Sent {}".format(image_names[i]))
            sleep(2)
