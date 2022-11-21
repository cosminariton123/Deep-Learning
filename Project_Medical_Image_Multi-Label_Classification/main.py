import tensorflow as tf

from data_loader import Generator
from model_generator import make_model
from history_plotting import plot_history
from preprocessing import preprocess_image

from paths import TRAIN_SAMPLES_DIR, TEST_SAMPLES_DIR, TRAIN_LABELS_PATH, VALIDATION_LABELS_PATH, VALIDATION_SAMPLES_DIR


def main():
    model = make_model()
    history = model.fit(
        Generator(samples_dir=TRAIN_SAMPLES_DIR,
        labels_path=TRAIN_LABELS_PATH,
        batch_size=128,
        preprocessing_procedure=preprocess_image,
        shuffle=True
        ),

        epochs = 100,

        validation_data = Generator(
            samples_dir=VALIDATION_SAMPLES_DIR,
            labels_path=VALIDATION_LABELS_PATH,
            batch_size=128,
            preprocessing_procedure=preprocess_image,
            shuffle=True
            ),

        shuffle = False
    )

    input("HAIDA")
    plot_history(history)


if __name__ == "__main__":
    main()
