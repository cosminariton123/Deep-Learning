import os

import tensorflow as tf
from keras.models import Sequential

from data_loader import TrainingGenerator
from paths import TRAIN_SAMPLES_DIR, TRAIN_LABELS_PATH, VALIDATION_LABELS_PATH, VALIDATION_SAMPLES_DIR
from preprocessing import preprocess_image
from history_plotting import plot_history
from config import BATCH_SIZE, EPOCHS

def search_for_best_model_and_save(model: Sequential , save_path):
    callbacks = tf.keras.callbacks.ModelCheckpoint(
                                    filepath = os.path.join(save_path ,"best_model.hdf5"),
                                    save_only_best_model = True
                                    )

    history = model.fit(
        TrainingGenerator(samples_dir=TRAIN_SAMPLES_DIR,
        labels_path=TRAIN_LABELS_PATH,
        batch_size=BATCH_SIZE,
        preprocessing_procedure=preprocess_image,
        shuffle=True
        ),

        epochs = EPOCHS,

        validation_data = TrainingGenerator(
            samples_dir=VALIDATION_SAMPLES_DIR,
            labels_path=VALIDATION_LABELS_PATH,
            batch_size=BATCH_SIZE,
            preprocessing_procedure=preprocess_image,
            shuffle=True
            ),

        callbacks = [callbacks],

        shuffle = False
    )

    plot_history(history, save_path)
