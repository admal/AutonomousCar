from log import log_info
from model.PredictModel import PredictModel

from utils import *


def main():

    (x_train, y_train), (x_val, y_val) = load_csv_data(
        "C:\\Users\\ASUS\\Documents\\PW\\SieciNeuronowe\\Projekt2\\Data", 0.1)

    x = load_batch(x_train, 0, len(x_train))
    x_v = load_batch(x_val, 0, len(x_val))

    log_info("START")
    model = PredictModel("C:\\Users\\ASUS\\Documents\\PW\\SieciNeuronowe\\Projekt2\\Model")
    model.load()
    tmp = x[28]
    tmpy = y_train[28]
    tmp_res = model.predict([x[28]])
    # model.train(x, y_train)
    # model.evaluate(x_v, y_val)

    log_info("Supposed: {}; Obtained: {}".format(tmpy, tmp_res[0]))


if __name__ == "__main__":
    main()