import os
import scipy
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 33, 100, 3


def convert_from_rgb(image):
    return image / 255


def resize(image):
    return scipy.misc.imresize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))


def crop(image):
    return image[60:-25, :, :]


def preprocess(image):
    image = crop(image)
    image = resize(image)
    image = convert_from_rgb(image)
    return image


def load_csv_data(learning_set_path, validation_set_size):
    data_df = pd.read_csv(os.path.join(os.getcwd(), learning_set_path, 'driving_log.csv'),
                          names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    # yay dataframes, we can select rows and columns by their names
    # we'll store the camera images as our input data
    X = data_df["center"].values
    # and our steering commands as our output data
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=validation_set_size, random_state=0)

    return (X_train, y_train), (X_valid, y_valid)


def load_batch(training_set, idx, batch_size):
    # input is 3 paths to images , output is expected steering_angle
    input_images_batch = []
    steering_angles_batch = []
    input_images_paths = training_set
    # for i in range(batch_size):
    input_images = []

    # load images for current iteration
    for img_path in input_images_paths:
        image = np.asfarray(Image.open(img_path))  # from PIL image to numpy array
        image = preprocess(image)  # apply the preprocessing
        input_images.append(image)

    # input_images_batch.append(input_images)
        # steering_angles_batch.append(steering_angles[idx + i])

    return input_images#, steering_angles_batch
