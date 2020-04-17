import requests
from kafka import KafkaConsumer, KafkaProducer
from json import loads
import numpy as np
from src.models.predict_model import predict
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
        message_value = message.value
        print(f"Processing topic key {message.key}")
        image_bytes = message_value['instances'][0]['image_bytes']
        image_np = np.array(image_bytes)
        image_np = np.array([image_np])  # add batch dimension since input of model is (batch_size, 32, 32, 3)
        assert image_np.shape == (1, 32, 32, 3)
        class_probs, prediction = predict(image_np)
        #TODO: run flask app and send predictions to the app
        prediction = labels[int(prediction)]
        print("Prediction probabilities:", class_probs)
        print("Prediction:", prediction)
        producer.send('imageClassifications', key=message.key, value=class_probs.tobytes())


