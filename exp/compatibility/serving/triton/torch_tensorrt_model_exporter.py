import os
from pathlib import Path

import torch
from torchvision.models import resnet50


version = 1
MODEL_DIR = Path(__file__).parent
export_path = MODEL_DIR / str(version)
export_path.mkdir(exist_ok=True, parents=True)
print('export_path = {}\n'.format(export_path))

# load model
model = resnet50(pretrained=True).eval()
# Generate a random input
inputs = torch.rand(1, 3, 224, 224)

# Generate a TorchScript model
traced_script_model = torch.jit.trace(model, inputs)

# Save the model
traced_script_model.save(os.path.join(export_path, "model.pt"))
