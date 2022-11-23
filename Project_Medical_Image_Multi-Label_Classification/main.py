import sys
import os
sys.path.append("../Deep-Learning")
from limit_gpu_memory_growth import limit_gpu_memory_growth

from model_generator import make_model
from tunning import search_for_best_model_and_save
from submission import load_and_make_submission
from paths import OUTPUT_DIR
    


def main():
    #limit_gpu_memory_growth()

    model = make_model()

    this_model_path = os.path.join(OUTPUT_DIR, "model0")
    if not os.path.exists(this_model_path):
        os.mkdir(this_model_path)
    search_for_best_model_and_save(model, this_model_path)

    load_and_make_submission(os.path.join(this_model_path, "best_model.hdf5"))


if __name__ == "__main__":
    main()
