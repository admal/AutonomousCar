from datetime import datetime, date, time
from log import log_info, log_error
from model.PredictModel import PredictModel
from model.TrainModel import TrainModel
from utils import *

import logging
logging.basicConfig(filename='training.log',level=logging.DEBUG)
MAX_ITERS = 40
# DATA_CSV_FILE="C:\\Users\\ASUS\\Documents\\PW\\SieciNeuronowe\\Projekt2\\Data"
# DATA_CSV_FILE="/home/auser/TrainingData"
# MODEL_DIRECTORY="/home/auser/Model"
# MODEL_DIRECTORY="C:\\Users\\ASUS\\Documents\\PW\\SieciNeuronowe\\Projekt2\\Model"
DATA_CSV_FILE="C:\\Studies\\AI\\driving_dataset"
MODEL_DIRECTORY="C:\\Studies\AI\\Model"


def main():
    (x_train, y_train), (x_val, y_val) = load_csv_data(DATA_CSV_FILE, 0.1)

    # x, y = load_batch(x_train, y_train, 0, len(y_train))
    # x = load_batch(x_train, 0, 1)
    # x_v, y_v = load_batch(x_val, y_val, 0, len(y_val))
    # x_v = load_batch(x_val, 0, 1)

    log_info("START")

    for i in range(0, MAX_ITERS):
        # load batch for each iteration, batches differ in chosen images and its mirrors resulting in different datasets
        x, y = load_batch(x_train, y_train, 0, len(y_train))
        x_v, y_v = load_batch(x_val, y_val, 0, len(y_val))
        iter_start_time = datetime.now()
        log_info("START ITERATION {}/{}".format(i+1, MAX_ITERS))
        model = TrainModel(MODEL_DIRECTORY)
        model.train(x, y)
        log_info("START: evaluation")
        model.evaluate(x_v, y_v)
        log_info("iteration last: {0:.3g} minutes".format((datetime.now() - iter_start_time).seconds / 60))
        print("iteration ", i+1, " last: {0:.3g} minutes".format((datetime.now() - iter_start_time).seconds / 60))

    log_info("FINISH")

    # log_info("Supposed: {}; Obtained: {}".format(tmpy, tmp_res[0]))


if __name__ == "__main__":
    main()