from datetime import datetime, date, time
import logging

IS_VERBOSE = True

def log_info(text):
    if IS_VERBOSE:
        logging.info(datetime.now().strftime("%d/%m/%Y %H:%M") + ": " + text)

def log_error(text):
    if IS_VERBOSE:
        logging.error(datetime.now().strftime("%d/%m/%Y %H:%M")+": " + text)