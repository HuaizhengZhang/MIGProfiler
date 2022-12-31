import numpy as np
import ray
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

# Initialize Ray
ray.init()

@ray.remote
def use_gpu():
    print(f'ray.get_gpu_ids(): {ray.get_gpu_ids()}')


# Create a remote function that loads the pre-trained ResNet50 model and makes a prediction on a given input tensor
@ray.remote(num_gpus=2)
def predict(input_np):
    tf.debugging.set_log_device_placement(True)

    # Convert inputs to tensor
    input_tensor = tf.convert_to_tensor(input_np)

    # Load the pre-trained ResNet50 model
    model = ResNet50(weights='imagenet')

    # Use the model to make a prediction on the input tensor
    predictions = model.predict(input_tensor).argmax(axis=1)

    # Return the predictions
    return predictions


# Check availalbe GPUs
ray.get(use_gpu.remote())

# Generate a random input numpy
random_np = np.random.randn(1, 224, 224, 3)

# Use Ray to run the prediction function on a GPU
predictions = ray.get(predict.remote(random_np))

# Print the predictions
print(predictions)
