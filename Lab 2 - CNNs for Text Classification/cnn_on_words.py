from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, BatchNormalization
from keras.layers import Embedding
from keras.datasets import imdb
import keras

from history_ploting import plot_history


def get_cnn_model_for_words(maxlen, max_features, embedding_dims, filters, ks, hidden_dims):

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

def main():
    maxlen = 1000
    max_features = 20000
    batch_size = 64
    embedding_dims = 100
    filters = 128
    ks = [3, 5]
    hidden_dims = 128
    epochs = 10


    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features, skip_top=20)

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    x_train = keras.utils.pad_sequences(x_train, maxlen = maxlen)
    x_test = keras.utils.pad_sequences(x_test, maxlen = maxlen)

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    model = get_cnn_model_for_words(maxlen, max_features, embedding_dims, filters, ks, hidden_dims)

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=1, shuffle=True)

    loss, accuracy = model.evaluate(x_train, y_train, verbose=True)
    print(f"Training Accuracy: {accuracy}")

    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    print(f"Testing Accuracy:  {accuracy}")
    plot_history(history)

if __name__ == "__main__":
    main()
