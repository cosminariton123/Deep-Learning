import keras
import os
import math
import numpy as np
import cv2


def load_samples(samples_dir):
    return [os.path.join(samples_dir, filepath) for filepath in os.listdir(samples_dir)]
    

def load_labels(path):
    with open(path, "r") as f:
        file = f.read()
    
    file = file.split("\n")
    file = file[1:]
    
    file = [[int(elem) for elem in line.split(",")[1:]] for line in file]

    return file


def shuffle_samples_and_labels(samples, labels):
    assert len(samples) == len(labels), "Length of samples should be the same to lenth of labels"

    permutation = np.arange(len(samples))
    permutation = np.random.shuffle(permutation)

    samples = samples[permutation]
    labels = labels[permutation]

    return samples, labels


class Generator(keras.utils.Sequence):
    def __init__(self, samples_dir, labels_path, batch_size, shuffle = True):
        self.image_paths = np.array(load_samples(samples_dir))

        self.labels = np.array(load_labels(labels_path))

        self.shuffle = shuffle

        if self.shuffle:
            self.image_paths, self.labels = shuffle_samples_and_labels(self.image_paths, self.labels)
        
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.image_paths) / self.batch_size)

    def __getitem__(self, iteration_n):
        filepaths = self.image_paths[self.batch_size * iteration_n : self.batch_size * (iteration_n + 1)]
        labels = self.labels[self.batch_size * iteration_n : self.batch_size * (iteration_n + 1)]
        
        return np.array([np.reshape(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), (64, 64, 1)) for filepath in filepaths]), labels

    def on_epoch_end(self):
        if self.shuffle:
            self.image_paths, self.labels = shuffle_samples_and_labels(self.image_paths, self.labels)
