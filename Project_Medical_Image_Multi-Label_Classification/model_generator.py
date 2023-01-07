import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPool2D

from custom_metrics import CUSTOM_METRICS
from config import INPUT_SIZE


def make_model():
    input = tf.keras.Input(shape = INPUT_SIZE)

    layer = input

    for _ in range(2):
        layer = Conv2D(128, (3, 3), activation="relu")(layer)
        layer = Dropout(0.8)(layer)
        layer = MaxPool2D()(layer)
    
    for _ in range(2):
        layer = Conv2D(64, (3, 3), activation="relu")(layer)
        layer = Dropout(0.6)(layer)

    layer = Flatten()(layer)

    for _ in range(2):
        layer = Dense(1024, activation="relu")(layer)
        layer = Dropout(0.5)(layer)

    layer = Dense(3, activation="sigmoid")(layer)

                
    model = tf.keras.Model(inputs = input, outputs = layer)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=CUSTOM_METRICS)

    model.summary()

    return model
