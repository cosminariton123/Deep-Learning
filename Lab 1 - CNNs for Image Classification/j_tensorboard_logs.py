import tensorflow as tf
import tensorflow_datasets as tfds
keras = tf.keras
import keras.layers as layers
import math
import os
import shutil


def make_model(input_shape):
    input = tf.keras.Input(shape = input_shape)
    layer = input

    for _ in range(2):
        layer = layers.Conv2D(64, (5, 5), activation = "relu")(layer)
        layer = layers.MaxPool2D((2, 2))(layer)
    layer = layers.Flatten()(layer)
    for u, a in zip([64, 10], ["relu", "softmax"]):
        layer = layers.Dense(u, activation = a)(layer)

    model = tf.keras.Model(inputs = input, outputs = layer)
    model.compile(
        optimizer = "adam",
        loss = "sparse_categorical_crossentropy",
        metrics = ["accuracy"]
    )
    
    return model


def train_with_logs(tensorboard_callback):
    batch_size = 64
    model = make_model((28, 28, 1))
    (train_ds, val_ds), meta = tfds.load(
                                        'mnist',
                                        as_supervised = True,
                                        with_info = True,
                                        split = [ 'train[:80%]', 'train[20%:]'],
                                        shuffle_files = True
                                        )
    n_samples = meta.splits['train'].num_examples

    test_ds, meta = tfds.load(
        "mnist",
        as_supervised = True,
        with_info = True,
    )
    test_ds = test_ds["test"]

    model.fit(
    train_ds.batch(batch_size).repeat(),
    steps_per_epoch = math.ceil(n_samples * 0.8 / batch_size),
    epochs = 5,
    validation_data = val_ds.batch(batch_size),
    validation_steps = math.ceil(n_samples * 0.2 / batch_size),
    shuffle = False,
    callbacks = [tensorboard_callback]
    )
    print(model.evaluate(test_ds.batch(batch_size)))


def main():
    log_dir = "./logs"

    batch_size = 64
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir = log_dir,
    histogram_freq = 0,
    batch_size = batch_size,
    write_graph = True,
    write_grads = False,
    write_images = False,
    embeddings_freq = 0,
    embeddings_layer_names =None,
    embeddings_metadata = None,
    embeddings_data = None,
    update_freq = 'batch'
    )
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    train_with_logs(tensorboard_callback)

if __name__ == "__main__":
    main()