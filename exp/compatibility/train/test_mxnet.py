import mxnet as mx
import numpy as np
from mxnet import gluon, nd, autograd
from mxnet.gluon.model_zoo.vision import resnet50_v1


def test_cuda_visibility():
    print('CUDA visibility')
    gpus = mx.test_utils.list_gpus()
    print('CUDA available', len(gpus) > 0)
    print('CUDA device count', mx.context.num_gpus())
    for gpu in gpus:
        print(f'*** CUDA {gpu} ***')
        # Create a context for the GPU
        memory = mx.context.gpu_memory_info(gpu)
        print(f'Memory free: {memory[0]}', f'Memory total: {memory[1]}')


def test_inference(device_num: int = 0):
    print(f'Inference on CUDA {device_num}')
    # Use the GPU device if available
    if mx.context.num_gpus() > 0:
        ctx = mx.gpu(device_num)
    else:
        ctx = mx.cpu()

    # Load the pre-trained MXNet ResNet50 model
    model = resnet50_v1(pretrained=True, ctx=ctx)
    # Generate a random input tensor on the specified context
    input_tensor = nd.array(np.random.randn(4, 3, 224, 224), ctx=ctx)
    # Conduct model prediction on the input tensor
    predictions = model(input_tensor).argmax(axis=1)
    # Print the predictions
    print(predictions)


def test_train():
    print('Training')
    ctx = mx.gpu(0)

    # Load the pre-trained ResNet50 model
    model = resnet50_v1(pretrained=True, ctx=ctx)

    optimizer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.1})
    criterion = gluon.loss.SoftmaxCrossEntropyLoss()

    for step in range(5):
        print(f'Step {step}')
        inputs = nd.array(np.random.rand(32, 3, 224, 224), ctx=ctx)
        labels = nd.array(np.random.randint(0, 1000, (32,)), ctx=ctx)

        with autograd.record():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step(batch_size=32)


def test_train_dp():
    pass


if __name__ == '__main__':
    test_cuda_visibility()
    test_inference()
    # test_inference(1)
    test_train()
