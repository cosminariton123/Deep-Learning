import tensorflow as tf
keras = tf.keras
import keras.layers as layers
import math
import os
import numpy as np
import cv2

def make_model():
    input = tf.keras.Input(shape = (28, 28, 1))
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
    def __init__(self, image_dir, batch_size, shuffle = True):
        self.image_paths = [os.path.join(image_dir, filepath) for filepath in os.listdir(image_dir)]
        self.shuffle = shuffle

        if self.shuffle:
            np.random.shuffle(self.image_paths)
        
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)

    def __getitem__(self, iteration_n):
        filepaths = self.image_paths[self.batch_size * iteration_n : self.batch_size * (iteration_n + 1)]
        
        return np.array([np.reshape(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), (28, 28, 1)) for filepath in filepaths]
        ), np.array([int(os.path.basename(filepath).split("_")[0]) for filepath in filepaths])

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_paths)



def main():

    model_with_shuffle = make_model()
    model_without_shuffle = make_model()

    model_with_shuffle.fit(
        Generator(os.path.join("custom_datasets", "mnist", "train"), batch_size = 64),
        epochs = 5,
        validation_data = Generator(os.path.join("custom_datasets", "mnist", "val"), batch_size = 64)
    )
    model_without_shuffle.fit(
        Generator(os.path.join("custom_datasets", "mnist", "train"), batch_size = 64, shuffle = False),
        epochs = 5,
        validation_data = Generator(os.path.join("custom_datasets", "mnist", "val"), batch_size = 64, shuffle = False)
    )

    print("Shuffle: ", model_with_shuffle.evaluate(Generator(os.path.join("custom_datasets", "mnist", "test"), batch_size = 64)))
    print("Without shuffle: ", model_without_shuffle.evaluate(Generator(os.path.join("custom_datasets", "mnist", "test"), batch_size = 64)))

if __name__ == "__main__":
    main()
