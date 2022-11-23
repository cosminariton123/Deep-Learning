import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPool2D


def make_model():
    input = tf.keras.Input(shape = (64, 64, 1))

    layer = input

    layer = Conv2D(64, (9, 9), activation="relu", input_shape=(64, 64, 1))(layer)
    layer = Dropout(0.5)(layer)
    layer = MaxPool2D()(layer)

    for _ in range(2):
        layer = Conv2D(64, (9, 9), activation="relu")(layer)
        layer = Dropout(0.5)(layer)

    layer = Flatten()(layer)

    for _ in range(2):
        layer = Dense(64, activation="relu")(layer)
        layer = Dropout(0.5)(layer)
        
    layer = Dense(3, activation="sigmoid")(layer)

                
    model = tf.keras.Model(inputs = input, outputs = layer)
                                                    #TODO CHANGE REQUIRED
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.BinaryAccuracy(name="mean F1 score", threshold=0.5)])

    model.summary()

    return model
