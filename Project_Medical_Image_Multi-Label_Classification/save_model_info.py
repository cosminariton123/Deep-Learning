import matplotlib.pyplot as plt
import os


def save_summary(string, save_path):
    with open(os.path.join(save_path, "Summary.txt"), "w") as f:
        f.write(string)


def create_custom_legend():
    plt.subplot(1, 2, 1)
    plt.legend(loc="center", bbox_to_anchor=(0.5, -0.25))
    plt.subplot(1, 2, 2)
    plt.legend(loc="center", bbox_to_anchor=(0.5, -0.25))


def create_custom_grid():
    plt.subplot(1, 2, 1)
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.grid(True)


def create_custom_title(fig, big_title):
    fig.suptitle(big_title, fontsize=16)
    plt.subplot(1, 2, 1)
    plt.title("Training")
    plt.subplot(1, 2, 2)
    plt.title("Validation")


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
    plt.savefig(os.path.join(path_to_save, "Loss.png"), bbox_inches="tight")


def plot_first_class_f1_score(history):
    first_class_f1_score = history.history["first_class_f1_score"]
    first_class_f1_score_validation = history.history["val_first_class_f1_score"]

    x = range(1, len(first_class_f1_score) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x, first_class_f1_score, color="goldenrod",  label="First class f1 score")
    plt.subplot(1, 2, 2)
    plt.plot(x, first_class_f1_score_validation, color="darkgoldenrod",  label="Validation first class f1 score")


def plot_second_class_f1_score(history):
    second_class_f1_score = history.history["second_class_f1_score"]
    second_class_f1_score_validation = history.history["val_second_class_f1_score"]

    x = range(1, len(second_class_f1_score) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x, second_class_f1_score, color="forestgreen",  label="Second class f1 score")
    plt.subplot(1, 2, 2)
    plt.plot(x, second_class_f1_score_validation, color="darkgreen",  label="Validation second class f1 score")


def plot_third_class_f1_score(history):
    third_class_f1_score = history.history["third_class_f1_score"]
    third_class_f1_score_validation = history.history["val_third_class_f1_score"]

    x = range(1, len(third_class_f1_score) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x, third_class_f1_score, color="red",  label="Third class f1 score")
    plt.subplot(1, 2, 2)
    plt.plot(x, third_class_f1_score_validation, color="magenta",  label="Validation third class f1 score")


def plot_mean_f1_score(history):
    mean_f1_score = history.history["mean_f1_score"]
    mean_f1_score_validation = history.history["val_mean_f1_score"]

    x = range(1, len(mean_f1_score) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x, mean_f1_score, color="cyan", label="Mean class f1 score")
    plt.subplot(1, 2, 2)
    plt.plot(x, mean_f1_score_validation, color="blue", label="Validation mean class f1 score")


def plot_f1_score(history, path_to_save):
    fig = plt.figure(figsize=(25, 5))
    create_custom_title(fig, "Training and validation f1 score")
    create_custom_grid()

    plot_first_class_f1_score(history)
    plot_second_class_f1_score(history)
    plot_third_class_f1_score(history)
    plot_mean_f1_score(history)

    create_custom_legend()
    plt.savefig(os.path.join(path_to_save, "F1 Score.png"), bbox_inches="tight")


def plot_first_class_precision(history):
    first_class_precision = history.history["first_class_precision"]
    first_class_precision_validation = history.history["val_first_class_precision"]

    x = range(1, len(first_class_precision) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x, first_class_precision, color="goldenrod",  label="First class precision")
    plt.subplot(1, 2, 2)
    plt.plot(x, first_class_precision_validation, color="darkgoldenrod",  label="Validation first class precision")


def plot_second_class_precision(history):
    second_class_precision = history.history["second_class_precision"]
    second_class_precision_validation = history.history["val_second_class_precision"]

    x = range(1, len(second_class_precision) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x, second_class_precision, color="forestgreen",  label="Second class precision")
    plt.subplot(1, 2, 2)
    plt.plot(x, second_class_precision_validation, color="darkgreen",  label="Validation second class precision")


def plot_third_class_precision(history):
    third_class_precision = history.history["third_class_precision"]
    third_class_precision_validation = history.history["val_third_class_precision"]

    x = range(1, len(third_class_precision) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x, third_class_precision, color="red",  label="Third class precision")
    plt.subplot(1, 2, 2)
    plt.plot(x, third_class_precision_validation, color="magenta",  label="Validation third class precision")


def plot_mean_precision(history):
    mean_precision = history.history["mean_precision"]
    mean_precision_validation = history.history["val_mean_precision"]

    x = range(1, len(mean_precision) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x, mean_precision, color="cyan", label="Mean class precision")
    plt.subplot(1, 2, 2)
    plt.plot(x, mean_precision_validation, color="blue", label="Validation mean class precision")


def plot_precision(history, path_to_save):
    fig = plt.figure(figsize=(25, 5))
    create_custom_title(fig, "Training and validation precision")
    create_custom_grid()

    plot_first_class_precision(history)
    plot_second_class_precision(history)
    plot_third_class_precision(history)
    plot_mean_precision(history)

    create_custom_legend()
    plt.savefig(os.path.join(path_to_save, "Precision.png"), bbox_inches="tight")


def plot_first_class_recall(history):
    first_class_recall = history.history["first_class_recall"]
    first_class_recall_validation = history.history["val_first_class_recall"]

    x = range(1, len(first_class_recall) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x, first_class_recall, color="goldenrod", label="First class recall")
    plt.subplot(1, 2, 2)
    plt.plot(x, first_class_recall_validation, color="darkgoldenrod", label="Validation first class recall")


def plot_second_class_recall(history):
    second_class_recall = history.history["second_class_recall"]
    second_class_recall_validation = history.history["val_second_class_recall"]

    x = range(1, len(second_class_recall) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x, second_class_recall, color="forestgreen", label="Second class recall")
    plt.subplot(1, 2, 2)
    plt.plot(x, second_class_recall_validation, color="darkgreen", label="Validation second class recall")


def plot_third_class_recall(history):
    third_class_recall = history.history["third_class_recall"]
    third_class_recall_validation = history.history["val_third_class_recall"]

    x = range(1, len(third_class_recall) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x, third_class_recall, color="red", label="Third class recall")
    plt.subplot(1, 2, 2)
    plt.plot(x, third_class_recall_validation, color="magenta", label="Validation third class recall")


def plot_mean_recall(history):
    mean_recall = history.history["mean_recall"]
    mean_recall_validation = history.history["val_mean_recall"]

    x = range(1, len(mean_recall) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x, mean_recall, color="cyan", label="Mean recall")
    plt.subplot(1, 2, 2)
    plt.plot(x, mean_recall_validation, color = "blue", label="Validation mean recall")


def plot_recall(history, path_to_save):
    fig = plt.figure(figsize=(25, 5))
    create_custom_title(fig, "Training and validation recall")
    create_custom_grid()

    plot_first_class_recall(history)
    plot_second_class_recall(history)
    plot_third_class_recall(history)
    plot_mean_recall(history)

    create_custom_legend()
    plt.savefig(os.path.join(path_to_save, "Recall.png"), bbox_inches="tight")


def plot_first_class_accuracy(history):
    first_class_accuracy = history.history["first_class_accuracy"]
    first_class_accuracy_validation = history.history["val_first_class_accuracy"]

    x = range(1, len(first_class_accuracy) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x, first_class_accuracy, color="goldenrod",  label="First class accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(x, first_class_accuracy_validation, color="darkgoldenrod",  label="Validation first class accuracy")


def plot_second_class_accuracy(history):
    second_class_accuracy = history.history["second_class_accuracy"]
    second_class_accuracy_validation = history.history["val_second_class_accuracy"]

    x = range(1, len(second_class_accuracy) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x, second_class_accuracy, color="forestgreen",  label="Second class accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(x, second_class_accuracy_validation, color="darkgreen",  label="Validation second class accuracy")


def plot_third_class_accuracy(history):
    third_class_accuracy = history.history["third_class_accuracy"]
    third_class_accuracy_validation = history.history["val_third_class_accuracy"]

    x = range(1, len(third_class_accuracy) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x, third_class_accuracy, color="red",  label="Third class accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(x, third_class_accuracy_validation, color="magenta",  label="Validation third class accuracy")


def plot_mean_accuracy(history):
    mean_accuracy = history.history["mean_accuracy"]
    mean_accuracy_validation = history.history["val_mean_accuracy"]

    x = range(1, len(mean_accuracy) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(x, mean_accuracy, color="cyan", label="Mean class accuracy")
    plt.subplot(1, 2, 2)
    plt.plot(x, mean_accuracy_validation, color="blue", label="Validation mean class accuracy")


def plot_accuracy(history, path_to_save):
    fig = plt.figure(figsize=(25, 5))
    create_custom_title(fig, "Training and validation accuracy")
    create_custom_grid()

    plot_first_class_accuracy(history)
    plot_second_class_accuracy(history)
    plot_third_class_accuracy(history)
    plot_mean_accuracy(history)

    create_custom_legend()
    plt.savefig(os.path.join(path_to_save, "Accuracy.png"), bbox_inches="tight")



def plot_history(history, path_to_save):
    plot_loss(history, path_to_save)
    plot_f1_score(history, path_to_save)
    plot_accuracy(history, path_to_save)
    plot_precision(history, path_to_save)
    plot_recall(history, path_to_save)
