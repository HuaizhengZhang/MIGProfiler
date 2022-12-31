# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors,
# and stored with the default serving key
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import ResNet50


version = 1
MODEL_DIR = Path(__file__).parent
export_path = os.path.join(MODEL_DIR, str(version))
print('export_path = {}\n'.format(export_path))

model = ResNet50(weights='imagenet')
tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

print(f'The model is exported at {export_path}')
