from keras.datasets import cifar10
from keras.utils import to_categorical
import tensorflow as tf
import keras.layers as layers
import matplotlib.pyplot as plt

def summarize_diagnostics(history):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test') 

    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    plt.show()


def preprocess_data(data):
    data = data.astype('float32')
    data = (data / 255.0)

    return data



def make_model():
    input = tf.keras.Input(shape = (32, 32, 3))
    layer = input

    layer = layers.Conv2D(32, (3, 3), activation = "relu", padding="same")(layer)
    layer = layers.Conv2D(32, (3, 3), activation = "relu", padding="same")(layer)
    layer = layers.MaxPool2D((2, 2))(layer)#try_without
    layer = layers.Dropout(1/4)(layer)
    layer = layers.Flatten()(layer)
    layer = layers.Dense(512, activation="relu")(layer)
    layer = layers.Dropout(1/2)(layer)
    layer = layers.Dense(10, activation="softmax")(layer)

    model = tf.keras.Model(inputs = input, outputs = layer)

    model.compile(
        optimizer = "adam",
        loss = "categorical_crossentropy",
        metrics = ["accuracy"]
    )
    return model

def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)
    
    return X_train, y_train, X_test, y_test


def main():
    model = make_model()
    X_train, y_train, X_test, y_test = load_data()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs = 20,
        shuffle=True
    )
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print('> %.3f' % (acc * 100.0))
    summarize_diagnostics(history)

if __name__ == "__main__":
    main()
