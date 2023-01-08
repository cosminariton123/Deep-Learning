import matplotlib.pyplot as plt
import os


def save_summary(string, save_path):
    with open(os.path.join(save_path, "Summary.txt"), "w") as f:
        f.write(string)


def plot_loss(history, path_to_save):
    training_loss = history.history["loss"]
    validation_loss = history.history["val_loss"]

    x = range(1, len(training_loss) + 1)

    plt.figure()
    plt.grid(True)

    plt.plot(x, training_loss, color="blue", label="Training loss")
    plt.plot(x, validation_loss, color="red", label="Validation loss")
    
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig(os.path.join(path_to_save, "Learning loss.png"), bbox_inches="tight")

def plot_accuracy(history, path_to_save):
    training_accuracy = history.history["mean_f1_score"]
    validation_accuracy = history.history["val_mean_f1_score"]

    x = range(1, len(training_accuracy) + 1)

    plt.figure()
    plt.grid(True)

    plt.plot(x, training_accuracy, color="blue", label="Training mean F1 Score")
    plt.plot(x, validation_accuracy, color="red", label="Validation mean F1 score")
    
    plt.title("Training and validation mean F1 score")
    plt.legend()
    plt.savefig(os.path.join(path_to_save, "Learning F1 Score.png"), bbox_inches="tight")


def plot_first_class_precision(history):
    first_class_precision = history.history["first_class_precision"]
    first_class_precision_validation = history.history["val_first_class_precision"]

    x = range(1, len(first_class_precision) + 1)

    plt.plot(x, first_class_precision, color="goldenrod",  label="First class precision")
    plt.plot(x, first_class_precision_validation, color="darkgoldenrod",  label="Validation first class precision")


def plot_second_class_precision(history):
    first_class_precision = history.history["second_class_precision"]
    first_class_precision_validation = history.history["val_second_class_precision"]

    x = range(1, len(first_class_precision) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x, first_class_precision, color="forestgreen",  label="Second class precision")
    plt.subplot(1, 2, 2)
    plt.plot(x, first_class_precision_validation, color="darkgreen",  label="Validation second class precision")


def plot_third_class_precision(history):
    first_class_precision = history.history["third_class_precision"]
    first_class_precision_validation = history.history["val_third_class_precision"]

    x = range(1, len(first_class_precision) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x, first_class_precision, color="red",  label="Third class precision")
    plt.subplot(1, 2, 2)
    plt.plot(x, first_class_precision_validation, color="magenta",  label="Validation third class precision")


def plot_average_precision(history):
    first_class_precision = history.history["average_class_precision"]
    first_class_precision_validation = history.history["val_average_class_precision"]

    x = range(1, len(first_class_precision) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x, first_class_precision, color="cyan", label="Average class precision")
    plt.subplot(1, 2, 2)
    plt.plot(x, first_class_precision_validation, color="blue", label="Validation average class precision")


def plot_precision(history, path_to_save):
    plt.figure(figsize=(25, 5))
    plt.subplot(1, 2, 1)
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.grid(True)

    plot_first_class_precision(history)
    plot_second_class_precision(history)
    plot_third_class_precision(history)
    plot_average_precision(history)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(path_to_save, "Precision.png"), bbox_inches="tight")


def plot_history(history, path_to_save):
    plot_loss(history, path_to_save)
    plot_accuracy(history, path_to_save)
    plot_precision(history, path_to_save)
