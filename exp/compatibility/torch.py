import torch

# Load the pre-trained ResNet50 model
model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)

# Generate a random input tensor
input = torch.randn(1, 3, 224, 224)

# Conduct model prediction
output = model(input)

# Print the predicted class index
print(output.argmax().item())