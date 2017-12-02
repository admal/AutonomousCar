import datetime


def log_info(text):
    print("[INFO] at ", str(datetime.datetime.now()),": ", text)

def log_error(text):
    print("[ERROR] at ", str(datetime.datetime.now()),": ", text)