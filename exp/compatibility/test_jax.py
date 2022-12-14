import jax
import jax.numpy as np
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax
from jaxlib.xla_extension import Device

# Use the GPU device if available
if jax.host_id() == 0:
    print(jax.devices()[0])

# Load the pre-trained JAX ResNet50 model
_, params = stax.serial(
    Dense(1024), Relu,
    Dense(1024), Relu,
    Dense(1024), Relu,
    Dense(10), LogSoftmax
)

# Generate a random input tensor
input_tensor = np.random.randn(1, 784)

# Conduct model prediction on the input tensor
predictions = stax.logsoftmax(stax.serial(params)(input_tensor))

# Print the predictions
print(predictions)
