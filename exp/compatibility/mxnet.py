import mxnet as mx
import numpy as np

# Use the GPU device if available
if mx.context.num_gpus() > 0:
    ctx = mx.gpu(1)
else:
    ctx = mx.cpu()

# Load the pre-trained MXNet ResNet50 model
model = mx.gluon.model_zoo.vision.resnet50_v1(pretrained=True)

# Generate a random input tensor on the specified context
input_tensor = mx.nd.array(np.random.randn(1, 3, 224, 224), ctx=ctx)

# Conduct model prediction on the input tensor
predictions = model(input_tensor).asnumpy()

# Print the predictions
print(predictions)
