import os

import tensorflow as tf

from data_loader import load_samples
from data_loader import PredictionsGenerator
from paths import TEST_SAMPLES_DIR, OUTPUT_DIR
from preprocessing import preprocess_image

def load_and_make_submission(model_path):
    model = tf.keras.models.load_model(model_path)

    predicts = model.predict(
        PredictionsGenerator(
            samples_dir=TEST_SAMPLES_DIR,
            batch_size=128,
            preprocessing_procedure=preprocess_image,
            shuffle=False
        )
    )


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
