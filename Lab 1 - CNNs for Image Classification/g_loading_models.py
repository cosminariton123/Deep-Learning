import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import random


def load_best_model(): 
    return tf.keras.models.load_model('./Lab 1 - CNNs for Image Classification/checkpoints/best_model.hdf5')


def predict_1_random(model):
    ds, meta = tfds.load('cifar10', as_supervised = True, with_info = True)
    test_ds = ds["test"]
    ds_as_numpy = tfds.as_numpy(test_ds)

    random_image, label = random.choice(list(ds_as_numpy))
    plt.title(label)
    plt.imshow(random_image)
    plt.show()

    im = np.float32(
    np.reshape(
                random_image,
                [1, 32, 32, 3],
                )
            )
    print("Prediction: ", np.argmax(model.predict(im), axis = -1), " Ground truth: ", label)


def predict_test_dataset(model):
    ds, meta = tfds.load('cifar10', as_supervised = True, with_info = True)
    test_ds = ds['test']
    model.evaluate(test_ds.batch(64))

def main():
    model = load_best_model()
    predict_1_random(model)
    predict_test_dataset(model)

if __name__ == '__main__':
    main()
