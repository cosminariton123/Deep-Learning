import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization


def make_model():
    model = Sequential()

    model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
    model.add(Dropout(0.4))
    model.add(Conv1D(filters, ks[0], 1, "same", activation="relu"))
    model.add(MaxPooling1D())
    model.add(Conv1D(filters, ks[1], 1, "same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(hidden_dims, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model
