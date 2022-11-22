import os

DATASET_PATH = os.path.join("Project_Medical_Image_Multi-Label_Classification", "dataset")

TRAIN_SAMPLES_DIR = os.path.join(DATASET_PATH, "train_images")

VALIDATION_SAMPLES_DIR = os.path.join(DATASET_PATH, "val_images")

TRAIN_LABELS_PATH = os.path.join(DATASET_PATH, "train_labels.csv")

VALIDATION_LABELS_PATH = os.path.join(DATASET_PATH, "val_labels.csv")

TEST_SAMPLES_DIR = os.path.join(DATASET_PATH, "test_images")

OUTPUT_DIR = os.path.join(DATASET_PATH, "output")