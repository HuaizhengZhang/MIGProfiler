import traceback

import torch
from torch import nn


def test_cuda_visibility():
    # CUDA visibility
    print('CUDA visibility')
    print('CUDA available', torch.cuda.is_available())
    print('CUDA device count', torch.cuda.device_count())
    device_count = torch.cuda.device_count()
    for device_id in range(device_count):
        print(f'*** CUDA {device_id} ***')
        print(torch.cuda.get_device_name(f'cuda:{device_id}'))
        print(torch.cuda.get_device_properties(f'cuda:{device_id}'))


def test_inference():
    from torchvision.models import resnet50, ResNet50_Weights
    # Load the pre-trained ResNet50 model
    model = resnet50(weights=ResNet50_Weights.DEFAULT).cuda()

    with torch.no_grad():
        # Generate a random input tensor
        inputs = torch.randn(4, 3, 224, 224).cuda()
        # Conduct model prediction
        outputs = model(inputs)
        # Print the predicted class index
        print(outputs.argmax(dim=1))


def test_multi_gpu_inference():
    from torchvision.models import resnet50, ResNet50_Weights
    # Load the pre-trained ResNet50 model
    model0 = resnet50(weights=ResNet50_Weights.DEFAULT).to('cuda:0')

    with torch.no_grad():
        print('Inference on CUDA 0')
        # Generate a random input tensor
        inputs0 = torch.randn(4, 3, 224, 224).to('cuda:0')
        # Conduct model prediction
        outputs0 = model0(inputs0)
        # Print the predicted class index
        print(outputs0.argmax(dim=1))

        try:
            print('Inference on CUDA 1')
            model1 = resnet50(weights=ResNet50_Weights.DEFAULT).to('cuda:1')
            # Generate a random input tensor
            inputs1 = torch.randn(4, 3, 224, 224).to('cuda:1')
            # Conduct model prediction
            outputs1 = model1(inputs1)
            # Print the predicted class index
            print(outputs1.argmax(dim=1))
        except RuntimeError:
            print(traceback.format_exc())


def test_train():
    from torchvision.models import resnet50, ResNet50_Weights
    # Load the pre-trained ResNet50 model
    model = resnet50(weights=ResNet50_Weights.DEFAULT).cuda()
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    criterion = nn.CrossEntropyLoss().cuda()

    for step in range(5):
        inputs = torch.randn(32, 3, 224, 224).cuda()
        labels = torch.randint(0, 1000, (32,)).cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_train_dp():
    from torchvision.models import resnet50, ResNet50_Weights

    # Assume we have a model and some data
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Use multiple GPUs
    model = nn.DataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    criterion = nn.CrossEntropyLoss().cuda()

    # Send the model to the GPUs
    model = model.cuda()

    # Train the model
    for step in range(10):
        # Send the data to the GPUs
        inputs = torch.randn(32, 3, 224, 224).cuda()
        labels = torch.randint(0, 1000, (32,)).cuda()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # Update the model parameters
        optimizer.zero_grad()
        optimizer.step()


if __name__ == '__main__':
    test_cuda_visibility()
    test_inference()
    test_multi_gpu_inference()
    test_train()
    test_train_dp()
