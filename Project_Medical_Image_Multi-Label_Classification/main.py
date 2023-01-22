import sys
import os
sys.path.append("../Deep-Learning")
from limit_gpu_memory_growth import limit_gpu_memory_growth

from model_generator import make_model, make_model_2
from tunning import search_for_best_model_and_save
from submission import load_and_make_submission
from paths import OUTPUT_DIR
    


def main():
    #limit_gpu_memory_growth()

    #Set INPUT_SIZE to (64, 64, 1) in config
    #model = make_model()

    #Set INPUT_SIZE to (64, 64) in config
    model = make_model_2()

    this_model_path = os.path.join(OUTPUT_DIR, "modelRNN")

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    if not os.path.exists(this_model_path):
        os.mkdir(this_model_path)
    search_for_best_model_and_save(model, this_model_path)

    #load_and_make_submission(os.path.join(OUTPUT_DIR, "modelCNN", "best_model.hdf5"))


if __name__ == "__main__":
    main()
