# Import necessary modules
import tensorflow as tf
from tensorflow.keras.applications import ResNet50


def test_cuda_visibility():
    print('CUDA visibility')
    print('CUDA available', len(tf.config.list_physical_devices('GPU')) > 0)
    gpu_devices = tf.config.experimental.get_visible_devices('GPU')
    print('CUDA device count', len(gpu_devices))
    for gpu_device in gpu_devices:
        # Print the properties of each GPU
        print(f'*** CUDA {gpu_device.name} ***')
        print('Device details:', tf.config.experimental.get_device_details(gpu_device))


def test_inference(device_num: int = 0):
    # Configure the GPU options to use only device 1
    gpus = tf.config.experimental.get_visible_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    with tf.device(f'/GPU:{device_num}'):
        # Load the pre-trained ResNet50 model
        model = ResNet50(weights='imagenet')
        # Generate a random tensor as input
        input_tensor = tf.random.uniform((4, 224, 224, 3))
        print(input_tensor.device)
        # Use the model to conduct prediction on the input tensor
        predictions = model.predict(input_tensor).argmax(axis=1)
        # Print the predictions
        print(predictions)


def test_train():
    tf.debugging.set_log_device_placement(True)

    x = tf.random.uniform((5 * 32, 224, 224, 3))
    y = tf.random.uniform((5 * 32, 1000), maxval=1, dtype=tf.dtypes.int64)

    model = ResNet50(weights=None)
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    with tf.device('/GPU:0'):
        # Fit the model on the input data and labels
        model.fit(x, y, epochs=1)


def test_train_dp():
    # Load the ResNet50 model
    model = ResNet50(weights=None)

    x = tf.random.uniform((10 * 32, 224, 224, 3))
    y = tf.random.uniform((10 * 32, 1000), maxval=1, dtype=tf.dtypes.int64)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Define a dataset and iterator
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(32).repeat()
    iterator = iter(dataset)

    # Define a strategy for data parallelism
    strategy = tf.distribute.MirroredStrategy()

    # Define a function for the data parallelism
    def data_parallelism(inputs, labels):
        # Calculate the predictions on each GPU
        predictions = []
        for i in range(strategy.num_replicas_in_sync):
            with strategy.scope():
                output = model(inputs[i], training=True)
                predictions.append(output)

        # Concatenate the predictions and calculate the loss
        predictions = tf.concat(predictions, axis=0)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, predictions))

        # Calculate the gradients and apply them to the model
        gradients = tf.gradients(loss, model.trainable_variables)
        optimizer = tf.optimizers.Adam(learning_rate=0.001)
        update_ops = optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Return the loss and update operations
        return loss, update_ops

    # Define a training loop
    for epoch in range(1):
        # Split the inputs and labels for each GPU
        inputs, labels = next(iterator)
        inputs = tf.split(inputs, strategy.num_replicas_in_sync)
        labels = tf.split(labels, strategy.num_replicas_in_sync)

    # Run the data parallelism function
    loss, update_ops = data_parallelism(inputs, labels)
    with tf.control_dependencies(update_ops):
        train_op = tf.no_op()
    with strategy.scope():
        train_op.run()


if __name__ == '__main__':
    test_cuda_visibility()
    # test_inference()
    # test_multi_gpu_inference()
    # test_train()
    test_train_dp()
