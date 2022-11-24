import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPool2D
from custom_metrics import mean_f1_score


def make_model():
    input = tf.keras.Input(shape = (64, 64, 1))

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
                                                    #TODO CHANGE REQUIRED
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[mean_f1_score])

    model.summary()

    return model
