import tensorflow as tf
import tensorflow_datasets as tfds
keras = tf.keras
import keras.layers as layers
import math


def make_model(input_shape, custom_metric):
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
        metrics = [custom_metric]
    )
    
    return model


def train_with_custom_metrics(custom_metric):
    batch_size = 64
    model = make_model((28, 28, 1), custom_metric)
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
    epochs = 1,
    validation_data = val_ds.batch(batch_size),
    validation_steps = math.ceil(n_samples * 0.2 / batch_size),
    shuffle = False,
    )
    print(model.evaluate(test_ds.batch(batch_size)))


def custom_metric(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_max(y_pred, axis = -1))


def ex1(y_true, y_pred):
    distance = 0.0
    for i in range(len(y_pred)):
        correct_class = tf.cast(y_true[i, 0], tf.int32)
        correct_activation = y_pred[i, correct_class]
        rest = tf.concat([y_pred[i, : correct_class], y_pred[i, correct_class + 1: ]], axis = 0)
        maximum_activation = tf.reduce_max(rest)
        distance += tf.abs(tf.cast(maximum_activation, tf.float32) - tf.cast(correct_activation, tf.float32))
    return distance / tf.cast(len(y_pred), tf.float32)


def ex2(y_true, y_pred):
    distance = 0.0
    for i in range(len(y_pred)):
        correct_class = tf.cast(y_true[i, 0], tf.int32)
        correct_activation = y_pred[i, correct_class]
        rest = tf.concat([y_pred[i, : correct_class], y_pred[i, correct_class + 1: ]], axis = 0)
        distance += tf.abs(tf.reduce_mean(rest) - tf.cast(correct_activation, tf.float32))
    return distance / tf.cast(len(y_pred), tf.float32)

def ex3(y_true, y_pred):
    distance = 0.0
    for i in range(len(y_pred)):
        correct_class = tf.cast(y_true[i, 0], tf.int32)
        correct_activation = y_pred[i, correct_class]
        rest = tf.concat([y_pred[i, : correct_class], y_pred[i, correct_class + 1: ]], axis = 0)
        aux_distance = tf.abs(tf.reduce_mean(rest) - tf.cast(correct_activation, tf.float32))
        if aux_distance > 0.9:
            distance += aux_distance
    return distance / tf.cast(len(y_pred), tf.float32)


def main():
    train_with_custom_metrics(ex1)
    train_with_custom_metrics(ex2)
    train_with_custom_metrics(ex3)


if __name__ == "__main__":
    main()

