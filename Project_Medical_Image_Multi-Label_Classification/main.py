import sys
import os
sys.path.append("../Deep-Learning")
from limit_gpu_memory_growth import limit_gpu_memory_growth

import tensorflow as tf
from data_loader import TrainingGenerator, PredictionsGenerator
from model_generator import make_model
from history_plotting import plot_history
from preprocessing import preprocess_image

from paths import TRAIN_SAMPLES_DIR, TEST_SAMPLES_DIR, TRAIN_LABELS_PATH, VALIDATION_LABELS_PATH, VALIDATION_SAMPLES_DIR, OUTPUT_DIR


def main():
    limit_gpu_memory_growth()

    model = make_model()

    callbacks = tf.keras.callbacks.ModelCheckpoint(
                                        filepath = os.path.join(OUTPUT_DIR ,"best_model.hdf5"),
                                        save_only_best_model = True
                                        )

    history = model.fit(
        TrainingGenerator(samples_dir=TRAIN_SAMPLES_DIR,
        labels_path=TRAIN_LABELS_PATH,
        batch_size=128,
        preprocessing_procedure=preprocess_image,
        shuffle=True
        ),

        epochs = 10,

        validation_data = TrainingGenerator(
            samples_dir=VALIDATION_SAMPLES_DIR,
            labels_path=VALIDATION_LABELS_PATH,
            batch_size=128,
            preprocessing_procedure=preprocess_image,
            shuffle=True
            ),

        callbacks = [callbacks],

        shuffle = False
    )

    input("Ready for results?")

    print("\n\nResults on test data:")
    print(model.evaluate(
        TrainingGenerator(
            samples_dir=VALIDATION_SAMPLES_DIR,
            labels_path=VALIDATION_LABELS_PATH,
            batch_size=128,
            preprocessing_procedure=preprocess_image,
            shuffle=True
            )))
    plot_history(history)

    predicts = model.predict(
        PredictionsGenerator(
            samples_dir=TEST_SAMPLES_DIR,
            batch_size=128,
            preprocessing_procedure=preprocess_image,
            shuffle=True
        )
    )


    from data_loader import load_samples

    ids = load_samples(TEST_SAMPLES_DIR)
    ids = [os.path.basename(elem) for elem in ids]

    result = "id,label1,label2,label3\n"

    for id, prediction in zip(ids ,predicts):
        prediction = [0 if x < 0.5 else 1 for x in prediction]
        
        prediction_as_string = ""
        for elem in prediction:
            prediction_as_string += f"{elem},"
        prediction_as_string = prediction_as_string[:-1]

        result += f"{id},{prediction_as_string}\n"

    with open(os.path.join(OUTPUT_DIR, "test_submission.csv"), "w") as f:
        f.write(result)


if __name__ == "__main__":
    main()
