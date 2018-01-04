from datetime import datetime, date, time
from log import log_info
from model3.TrainModel import TrainModel
from utils import *
import logging
from config import *
import tensorflow as tf


logging.basicConfig(filename=TRAIN_LOG_FILE, level=logging.DEBUG)


def main():
    (x_train, y_train), (x_val, y_val) = load_csv_data(DATA_CSV_FILE, VALIDATION_SET_SIZE)

    log_info("START")
    
    model = TrainModel() # this could have parameter MODEL_DIRECTORY
    model.train(x_train, x_val, y_train, y_val) #this could be repeated for N iterations

    log_info("FINISH")

if __name__ == "__main__":
    main()