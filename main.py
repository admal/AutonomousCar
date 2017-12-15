from log import log_info, log_error
from model.PredictModel import PredictModel
from model.TrainModel import TrainModel
from utils import *

import logging
logging.basicConfig(filename='training.log',level=logging.DEBUG)

def main():
    (x_train, y_train), (x_val, y_val) = load_csv_data(
        "/home/auser/TrainingData", 0.1)

    x = load_batch(x_train, 0, len(x_train))
    x_v = load_batch(x_val, 0, len(x_val))

    log_info("START")

    iters = 100
    for i in range(0, iters):
        log_info("START ITERATION {}/{}".format(i, iters))
        model = TrainModel("/home/auser/Model")
        model.train(x, y_train)
        log_info("START: evaluation")
        model.evaluate(x_v, y_val)

    log_info("FINISH")

    # log_info("Supposed: {}; Obtained: {}".format(tmpy, tmp_res[0]))


if __name__ == "__main__":
    main()