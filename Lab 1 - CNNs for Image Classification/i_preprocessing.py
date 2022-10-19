import tensorflow as tf
import tensorflow_datasets as tfds
keras = tf.keras
import keras.layers as layers
import math
import os
import numpy as np
import cv2


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



class Generator(keras.utils.Sequence):
    def __init__(self, image_dir, batch_size, shuffle = True, preprocessing_function = None):
        self.image_paths = [os.path.join(image_dir, filepath) for filepath in os.listdir(image_dir)]
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.preprocessing_function = preprocessing_function

        if self.shuffle:
            np.random.shuffle(self.image_paths)

    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)

    def __getitem__(self, iteration_n):
        filepaths = self.image_paths[self.batch_size * iteration_n : self.batch_size * (iteration_n + 1)]
        
        data, labels = np.array([np.reshape(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), (28, 28, 1)) for filepath in filepaths]
        ), np.array([int(os.path.basename(filepath).split("_")[0]) for filepath in filepaths])

        preprocessed_data = list()
        preprocessed_labels = list()
        for elem_data, elem_label in zip(data, labels):
            preprocessed_elem_data, preprocessed_elem_label = self.preprocessing_function(elem_data, elem_label)
            preprocessed_data.append(preprocessed_elem_data)
            preprocessed_labels.append(preprocessed_elem_label)
        preprocessed_data = np.array(preprocessed_data)
        preprocessed_labels = np.array(preprocessed_labels)

        return preprocessed_data, preprocessed_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_paths)



def format_image(image, label):
    image = tf.cast(image, tf.float32)
    image = image / 255.
    image = tf.image.resize(image, (128, 128))
    return image, label



def preprocessing_on_custom_dataset():
    batch_size = 64
    model = make_model((128, 128, 1))
    model.fit(
        Generator(os.path.join("custom_datasets", "mnist", "train"), batch_size = batch_size, shuffle = True, preprocessing_function = format_image),
        epochs = 5,
        validation_data = Generator(os.path.join("custom_datasets", "mnist", "val"), batch_size = batch_size, shuffle = False, preprocessing_function = format_image),
        shuffle = False
    )
    print(model.evaluate(Generator(os.path.join("custom_datasets", "mnist", "test"), batch_size = batch_size, shuffle = False, preprocessing_function = format_image)))


def preprocessing_on_tfds():
    batch_size = 64
    model = make_model((128, 128, 1))
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

    train_ds = train_ds.map(format_image)
    val_ds = val_ds.map(format_image)
    test_ds = test_ds.map(format_image)

    model.fit(
    train_ds.batch(batch_size).repeat(),
    steps_per_epoch = math.ceil(n_samples * 0.8 / batch_size),
    epochs = 5,
    validation_data = val_ds.batch(batch_size),
    validation_steps = math.ceil(n_samples * 0.2 / batch_size),
    shuffle = False,
    )
    print(model.evaluate(test_ds.batch(batch_size)))


def main():
    preprocessing_on_custom_dataset()
    preprocessing_on_tfds()

if __name__ == "__main__":
    main()
