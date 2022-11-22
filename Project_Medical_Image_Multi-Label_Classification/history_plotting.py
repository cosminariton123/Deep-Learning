import matplotlib.pyplot as plt
import os

def plot_history(history, path_to_save):
    plt.style.use("ggplot")
    acc = history.history["mean F1 score"]
    val_acc = history.history["val_mean F1 score"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, "b", label="Training mean F1 score")
    plt.plot(x, val_acc, "r", label="Validation mean F1 score")
    plt.title("Training and validation mean F1 score")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, "b", label="Training loss")
    plt.plot(x, val_loss, "r", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    
    plt.savefig(os.path.join(path_to_save, "Learning graph.png"), bbox_inches="tight")
