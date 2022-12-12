# Import necessary modules
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

# Set the desired TensorFlow output level for debugging
tf.debugging.set_log_device_placement(True)

# Configure the GPU options to use only device 1
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[1], True)
    except RuntimeError as e:
        print(e)

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Generate a random tensor as input
input_tensor = tf.random.uniform((1, 224, 224, 3))

# Use the model to conduct prediction on the input tensor
predictions = model.predict(input_tensor)

# Print the predictions
print(predictions)

