from keras.datasets import imdb
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer

import numpy as np

from history_ploting import plot_history

def map_ids_to_words(word_classification_dataset):
    word_to_id = word_classification_dataset.get_word_index()

    word_to_id = {key : (value + 3) for key, value in word_to_id.items()}

    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    word_to_id["<UNUSED>"] = 3

    id_to_word = {value: key for key, value in word_to_id.items()}

    return id_to_word


def restore_text_data(dataset_samples, dataset):
    str_data = list()

    id_to_word = map_ids_to_words(dataset)

    for i in range(len(dataset_samples)):
        str_data.append(" ".join(id_to_word[id] for id in dataset_samples[i]))


    return str_data


def get_vocabulary(x_train_str):
    return {chr for string in x_train_str for chr in string}

def create_tokenizer(x_train_str, chars):
    tk = Tokenizer(num_words=None, char_level=True, oov_token="UNK")

    tk.fit_on_texts(x_train_str)

    char_dict = {chr: idx + 1 for idx, chr in enumerate(chars)}

    tk.word_index = char_dict

    tk.word_index[tk.oov_token] = max(char_dict.values()) + 1

    return tk


def prerocess_dataset(dataset, tk: Tokenizer, maxlen):
    sequences = tk.texts_to_sequences(dataset)

    proc_data = keras.utils.pad_sequences(sequences, maxlen=maxlen, padding="post")
    proc_data = np.array(proc_data)
    
    return proc_data


def load_embedding_weights(tk: Tokenizer):

    embedding_weights = list()

    embedding_weights.append(np.zeros(len(tk.word_index)))

    for char, i in tk.word_index.items():
        onehot = np.zeros(len(tk.word_index))
        onehot[i - 1] = 1
        embedding_weights.append(onehot)

    embedding_weights = np.array(embedding_weights)

    return embedding_weights



def make_cnn_model_for_characters(vocab_size, input_size, embedding_weights, conv_layers, dropout_p, optimizer, loss):
    model = Sequential()

    model.add(Embedding(vocab_size + 1, vocab_size, input_length=input_size, weights=[embedding_weights]))

    for filter_num, filter_size, pooling_size in conv_layers:
        model.add(Conv1D(filter_num, filter_size, activation="relu"))
        
        if pooling_size != -1:
            model.add(MaxPooling1D(pool_size=pooling_size))

    model.add(Flatten())

    model.add(Dense(1024, activation="relu"))

    model.add(Dropout(dropout_p))

    model.add(Dense(2, activation="sigmoid"))

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    
    return model

def main():
    maxlen = 1024
    input_size = maxlen
    dropout_p = 0.5
    optimizer = 'adam'
    loss = "binary_crossentropy"
    conv_layers = [[128, 7, 3], [128, 5, -1], [128, 3, -1], [128, 3, 3]]
    batch_size = 128
    epochs = 10

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000, index_from=3)
    x_train_str = restore_text_data(x_train, imdb)
    x_test_str = restore_text_data(x_test, imdb)

    chars = get_vocabulary(x_train_str)
    tk = create_tokenizer(x_train_str, chars)

    train_data = prerocess_dataset(x_train_str, tk, maxlen)
    test_data = prerocess_dataset(x_test_str, tk, maxlen)

    vocab_size = len(tk.word_index)

    print(f"Vocabulary: {tk.word_index}")
    print(f"Vocabulary size: {len(tk.word_index)}")

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    embedding_weights = load_embedding_weights(tk)

    print(f"Embedding weights shape: {embedding_weights.shape}")
    print(f"Embedding weights\n: {embedding_weights}")

    model = make_cnn_model_for_characters(vocab_size, input_size, embedding_weights, conv_layers, dropout_p, optimizer, loss)
    history = model.fit(train_data, y_train, validation_data=(test_data, y_test), batch_size=batch_size, epochs=epochs, shuffle=True)

    plot_history(history)

if __name__ == "__main__":
    main()
