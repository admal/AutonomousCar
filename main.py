from datetime import datetime, date, time
from log import log_info, log_error
from model.PredictModel import PredictModel
from model.TrainModel import TrainModel
from utils import *

import logging
logging.basicConfig(filename='training.log',level=logging.DEBUG)
MAX_ITERS = 10
# DATA_CSV_FILE="C:\\Users\\ASUS\\Documents\\PW\\SieciNeuronowe\\Projekt2\\Data"
DATA_CSV_FILE="/home/auser/TrainingData"
MODEL_DIRECTORY="/home/auser/Model"
# MODEL_DIRECTORY="C:\\Users\\ASUS\\Documents\\PW\\SieciNeuronowe\\Projekt2\\Model"

def main():
    (x_train, y_train), (x_val, y_val) = load_csv_data(DATA_CSV_FILE, 0.1)

    x = load_batch(x_train, 0, len(x_train))
    x_v = load_batch(x_val, 0, len(x_val))

    log_info("START")

    for i in range(0, MAX_ITERS):
        iter_start_time = datetime.now()
        log_info("START ITERATION {}/{}".format(i+1, MAX_ITERS))
        model = TrainModel(MODEL_DIRECTORY)
        model.train(x, y_train)
        log_info("START: evaluation")
        model.evaluate(x_v, y_val)
        log_info("iteration last: {0:.3g} minutes".format((datetime.now() - iter_start_time).seconds / 60))
        # print("iteration ", i+1, " last: {0:.3g} minutes".format((datetime.now() - iter_start_time).seconds / 60))

    log_info("FINISH")

    # log_info("Supposed: {}; Obtained: {}".format(tmpy, tmp_res[0]))


if __name__ == "__main__":
    main()