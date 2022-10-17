import tensorflow as tf
import tensorflow_datasets as tfds
keras = tf.keras
import keras.layers as layers
import matplotlib.pyplot as plt
import math


def load_cifar_dataset():
    ds, meta = tfds.load("cifar10", as_supervised=True, with_info=True)
    train_ds = ds["train"]
    return train_ds, meta


def show_dataset(data):
    for image, label in tfds.as_numpy(data):
        plt.title(label)
        plt.imshow(image)
        plt.show()


def cnn_to_train_on_cifar():
    input = tf.keras.Input(shape = (32, 32, 3))
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

    batch_size = 64
    ds, meta = tfds.load("cifar10", as_supervised=True, with_info=True)
    train_ds = ds["train"]


    n_samples = meta.splits["train"].num_examples
    model.fit(
        train_ds.batch(batch_size).repeat(),
        epochs = 5,
        steps_per_epoch = math.ceil(n_samples / batch_size)
    )



def main():
    cnn_to_train_on_cifar()



if __name__ == "__main__":
    main()