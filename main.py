from datetime import datetime, date, time
from log import log_info, log_error
from model.PredictModel import PredictModel
from model.TrainModel import TrainModel
from utils import *

import logging
logging.basicConfig(filename='training.log',level=logging.DEBUG)

def main():
    (x_train, y_train), (x_val, y_val) = load_csv_data(
        "C:/Studies/AI/driving_dataset", 0.1)

    x, y = load_batch(x_train, y_train, 0, 20)
    x_v, y_v = load_batch(x_val, y_val, 0, 10)

    log_info("START")

    iters = 100

    for i in range(0, iters):
        iter_start_time = datetime.now()
        log_info("START ITERATION {}/{}".format(i, iters))
        model = TrainModel("C:/Studies/AI/Model")
        model.train(x, y)
        log_info("START: evaluation")
        model.evaluate(x_v, y_v)
        log_info("iteration last: {} minutes".format((iter_start_time - datetime.now()).seconds / 60))

    log_info("FINISH")

    # log_info("Supposed: {}; Obtained: {}".format(tmpy, tmp_res[0]))


if __name__ == "__main__":
    main()