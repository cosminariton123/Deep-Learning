import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization, Conv2D, MaxPool2D


def make_model():
    model = Sequential()

    model.add(Conv2D(64, (5,5), activation="relu", input_shape=(64, 64, 1)))
    model.add(Dropout(0.4))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(64, (5,5), activation="relu"))
    model.add(Dropout(0.4))
    model.add(MaxPool2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model
