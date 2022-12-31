import paddle
from paddle import nn
from paddle.vision.models import resnet50


def test_cuda_visibility():
    print('CUDA visibility')
    gpu_device_count = paddle.device.cuda.device_count()
    print('CUDA available', gpu_device_count > 0)
    print('CUDA device count', gpu_device_count)
    for gpu in range(gpu_device_count):
        print(f'*** CUDA {gpu} ***')
        print('Device Details:')
        print('Name', paddle.device.cuda.get_device_name(gpu))
        print('Capability', paddle.device.cuda.get_device_capability(gpu))
        print('Properties', paddle.device.cuda.get_device_properties(gpu))


def test_inference(device_num: int = 0):
    print(f'Inference on CUDA {device_num}')
    # Config PaddlePaddle global device
    paddle.set_device(f"gpu:{device_num}")
    # Load the pre-trained ResNet50 model
    model = resnet50(pretrained=True)
    # Generate a random input tensor
    input_tensor = paddle.uniform((4, 3, 224, 224))
    # Conduct model prediction on the input tensor
    predictions = model(input_tensor).argmax(axis=1)    
    # Print the predictions
    print(predictions, predictions.place)


def test_train():
    print('Training')
    # Config PaddlePaddle global device
    paddle.set_device('gpu:0')
    # Load the pre-trained ResNet50 model
    model = resnet50(pretrained=True)
    model.train()

    optimizer = paddle.optimizer.SGD(learning_rate=1e-1, parameters=model.parameters())
    criterion = nn.CrossEntropyLoss()

    for step in range(5):
        print(f'Step {step}')
        inputs = paddle.uniform((32, 3, 224, 224))
        labels = paddle.randint(0, 1000, (32,))

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    test_cuda_visibility()
    test_inference()
    # test_inference(1)
    test_train()
