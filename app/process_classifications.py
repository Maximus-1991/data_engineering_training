import requests
from kafka import KafkaConsumer, KafkaProducer
from json import loads, dumps, load
import numpy as np

from src.utils.dump_load_pickle import load_labels

labels = load_labels()

consumer = KafkaConsumer(
    'imageClassifications',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id='my-group',
    value_deserializer=lambda x: np.frombuffer(x)) #)


if __name__ == "__main__":
    for message in consumer:
        class_probs = message.value
        prediction = np.argmax(class_probs)
        prediction = labels[int(prediction)]
        print("Prediction:", prediction)
