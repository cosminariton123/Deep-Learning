import tensorflow as tf
import tensorflow_datasets as tfds
keras = tf.keras
import keras.layers as layers
import math
import os

def make_model():
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
    
    return model



def save_model_after_every_epoch(model):
    batch_size = 64
    (train_ds, val_ds), meta = tfds.load(
                                        'cifar10',
                                        as_supervised = True,
                                        with_info = True,
                                        split = [ 'train[:80%]', 'train[20%:]']
                                        )
    n_samples = meta.splits['train'].num_examples

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = './checkpoints/model.hdf5')

    model.fit(
    train_ds.batch(batch_size).repeat(),
    steps_per_epoch = math.ceil(n_samples * 0.8 / batch_size),
    epochs = 5,
    validation_data = val_ds.batch(batch_size),
    validation_steps = math.ceil(n_samples * 0.2 / batch_size),
    callbacks = [checkpoint_callback]
    )



def save_only_best_model(model):
    batch_size = 64
    (train_ds, val_ds), meta = tfds.load(
                                        'cifar10',
                                        as_supervised = True,
                                        with_info = True,
                                        split = [ 'train[:80%]', 'train[20%:]']
                                        )
    n_samples = meta.splits['train'].num_examples

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                                filepath = './checkpoints/best_model.hdf5',
                                                                save_only_best_model = True
                                                                )

    model.fit(
    train_ds.batch(batch_size).repeat(),
    steps_per_epoch = math.ceil(n_samples * 0.8 / batch_size),
    epochs = 5,
    validation_data = val_ds.batch(batch_size),
    validation_steps = math.ceil(n_samples * 0.2 / batch_size),
    callbacks = [checkpoint_callback]
    )



def main():
    model = make_model()
    save_model_after_every_epoch(model)
    save_only_best_model(model)



if __name__ == '__main__':
    main()
