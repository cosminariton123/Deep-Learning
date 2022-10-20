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

    # 3.2. Add the layers (check out the work done in the previous lab)

    ########################################################################################
    # 3.2.1. Add an embedding layer which maps our vocab indices (max_features) into embedding_dims dimensions
    model.add(Embedding(max_features, embedding_dims, input_length=maxlen))

    # TODO 3.2.2. Dropout with a probability of 0.4
    model.add(Dropout(0.4))

    # TODO 3.2.3. Add a Convolution1D layer, with 128 filters, kernel size ks[0], padding same, activation relu and stride 1
    model.add(Conv1D(filters = filters, kernel_size = ks[0], padding = "same", activation = "relu", strides = 1))

    # TODO 3.2.4. Use max pooling after the CONV layer
    model.add(MaxPooling1D())

    # TODO 3.2.5. Add a CONV layer, similar in properties to what we have above (3.2.3.) and kernel size 5
    model.add(Conv2D(filters = filters, kernel_size = ks[1], padding = "same", activation = "relu", strides = 1))

    # TODO 3.2.6. Add a Batch Normalization layer in order to reduce overfitting
    model.add(BatchNormalization())

    # TODO 3.2.7. Use max pooling again
    model.add(MaxPooling1D())

    # TODO 3.2.8. Add a flatten layer
    model.add(Flatten())

    # TODO 3.2.9. Add a dense layer with hidden_dims hidden units and activation relu
    model.add(Dense(hidden_dims = hidden_dims, activation = "relu"))

    # TODO 3.2.10. Add a dropout layer with a dropout probability of 0.5 
    model.add(Dropout(0.5))

    # TODO 3.2.11. We project onto a single unit output layer, and squash it with a sigmoid
    model.add(Dense(1, activation='sigmoid'))
    ##################################################################################

    # TODO 3.3. Compile the model
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
