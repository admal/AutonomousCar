from log import log_info, log_error
from model.PredictModel import PredictModel
from model.TrainModel import TrainModel
from utils import *

import logging
logging.basicConfig(filename='training.log',level=logging.DEBUG)

def main():

    logging.info("TEST")
    log_info("TEST log_info")
    log_error("Test log_error")
    return

    (x_train, y_train), (x_val, y_val) = load_csv_data(
        "/home/auser/TrainingData", 0.1)

    x = load_batch(x_train, 0, len(x_train))
    x_v = load_batch(x_val, 0, len(x_val))

    log_info("START")
    # model = PredictModel("C:\\Users\\ASUS\\Documents\\PW\\SieciNeuronowe\\Projekt2\\Model")
    # model.load()
    model = TrainModel("/home/auser/Model")

    # tmp = x[28]
    # tmpy = y_train[28]
    # tmp_res = model.predict([x[28]])
    model.train(x, y_train)
    model.evaluate(x_v, y_val)

    # log_info("Supposed: {}; Obtained: {}".format(tmpy, tmp_res[0]))


if __name__ == "__main__":
    main()