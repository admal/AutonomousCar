import datetime


def log_info(text, is_verbose = False):
    if is_verbose:
        print("[INFO] at ", str(datetime.datetime.now()),": ", text)

def log_error(text):
    print("[ERROR] at ", str(datetime.datetime.now()),": ", text)