from datetime import datetime, date, time
from log import log_info
from model.TrainModel import TrainModel
from utils import *
import logging
from config import *

logging.basicConfig(filename=TRAIN_LOG_FILE, level=logging.DEBUG)


def main():
    (x_train, y_train), (x_val, y_val) = load_csv_data(DATA_CSV_FILE, 0.1)

    # x, y = load_batch(x_train, y_train, 0, len(x_train))
    # x_v, y_v = load_batch(x_val, y_val, 0, len(x_val))
 
    log_info("START")

    for i in range(0, MAX_ITERS):
        # load batch for each iteration, batches differ in chosen images and its mirrors resulting in different datasets
        if BATCH_SIZE is None:
            x, y = load_batch(x_train, y_train, 0, len(y_train))
            x_v, y_v = load_batch(x_val, y_val, 0, len(y_val))
        else:
            x, y = load_batch(x_train, y_train, 0, BATCH_SIZE)
            x_v, y_v = load_batch(x_val, y_val, 0, BATCH_SIZE)
            
        iter_start_time = datetime.now()
        log_info("START ITERATION {}/{}".format(i+1, MAX_ITERS))
        model = TrainModel(MODEL_DIRECTORY)
        model.train(x, y)
        log_info("START: evaluation")
        model.evaluate(x_v, y_v)
        log_info("iteration last: {0:.3g} minutes".format((datetime.now() - iter_start_time).seconds / 60))
        print("iteration ", i+1, " last: {0:.3g} minutes".format((datetime.now() - iter_start_time).seconds / 60))

    log_info("FINISH")

if __name__ == "__main__":
    main()