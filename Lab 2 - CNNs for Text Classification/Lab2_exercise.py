from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding, Input
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Conv2D
from keras import metrics
import keras.layers as layers
import numpy as np

import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import keras



def get_cnn_model_v1():
    maxlen = 1000
    max_features = 10000
    batch_size = 64
    embedding_dims = 100
    filters = 128
    ks = [3, 5, 5] # kernel_size
    hidden_dims = 128
    epochs = 10

    model = Sequential()

    model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
    model.add(Dropout(0.4))
    model.add(Conv1D(128, 3, 1, "same", activation="relu"))
    model.add(MaxPooling1D())
    model.add(Conv1D(128, 5, 1, "same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model



def main():
    maxlen = 1000
    max_features = 10000
    batch_size = 64
    embedding_dims = 100
    filters = 128
    ks = [3, 5, 5] # kernel_size
    hidden_dims = 128
    epochs = 10


    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
    #x_train = x_train[20:]

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    x_train = keras.utils.pad_sequences(x_train, maxlen = maxlen)
    x_test = keras.utils.pad_sequences(x_test, maxlen = maxlen)

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)


  
    model = get_cnn_model_v1()

    # TODO 3.4. Train (fit) the model
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=1, shuffle=True)
        


if __name__ == "__main__":
    main()
