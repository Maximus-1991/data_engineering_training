import requests
from kafka import KafkaConsumer, KafkaProducer
from json import loads
import numpy as np
from src.utils.dump_load_pickle import load_labels

labels = load_labels()

consumer = KafkaConsumer(
    'newImages',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id='my-group',
    value_deserializer=lambda x: loads(x.decode('utf-8')))

producer = KafkaProducer(bootstrap_servers=['localhost:9092'])

if __name__ == "__main__":
    for message in consumer:
        headers = {"content-type": "application/json"}
        tf_serving_api_endpoint = "http://localhost:8501/v1/models/image_classification:predict"
        #tf_serving_api_endpoint = api_endpoint = f"http://{host}:{port}/v{model_version}/models/{model_name}:predict"
        r = requests.post(url=tf_serving_api_endpoint, json=message.value, headers=headers)
        class_probs = np.array(r.json()['predictions'][0])
        prediction = np.argmax(class_probs)
        prediction = labels[int(prediction)]
        print("Prediction probabilities:", class_probs)
        print("Prediction:", prediction)
        producer.send('imageClassifications', key=message.key, value=class_probs.tobytes())

