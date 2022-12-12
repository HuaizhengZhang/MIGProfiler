import jax
import jax.numpy as np
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax

# Use the GPU device if available
if jax.host_id() == 0:
    jax.devices()[1].use()

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