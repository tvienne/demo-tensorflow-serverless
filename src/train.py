import tensorflow as tf
import matplotlib.pyplot as plt
from sagemaker.predictor import json_serializer, csv_serializer, json_deserializer, Predictor

from sagemaker.deserializers import JSONLinesDeserializer
from sagemaker.serializers import JSONLinesSerializer
import tensorflow as tf
import numpy as np
import time

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# invoke the endpoint
endpoint_name="tensorflow-realtime-endpoint-2"
predictor = Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker_session,
    serializer=json_serializer,
    content_type="application/json",
    accept="application/json",
)

# make a simple prediction
observation = 2
predictions = predictor.predict(test_data[observation])
print("predictions observation %s :" % observation,  predictions)
print("Labels observation %s :" % observation,  test_labels[observation])

# for multiple observations
list_time_elapsed = []
observation = 2
for i in range(0, 10):
    start = time.time()
    predictions = predictor.predict(test_data[observation])
    prediction = predictions
    label = test_labels[observation]
    end = time.time()
    print(end - start)
    list_time_elapsed.append(end - start)

print(list_time_elapsed)
print("mean invokation time = %s seconds" % np.mean(list_time_elapsed))