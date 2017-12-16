import os
import scipy
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split

# IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 160, 320, 3
# IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 64, 256, 3


def convert_from_rgb(image):
    return (image / 127.5) - 1.0
    # return image / 255


def resize(image):
    return scipy.misc.imresize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))


def crop(image):
    return image[60:-25, :, :]


def preprocess(image):
    image = crop(image)
    image = resize(image)
    # cvImage = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow("window", cvImage)
    # while True:
    #     k = cv2.waitKey(0) & 0xFF
    #     if k == 27: break
    # return
    # cv2.destroyAllWindows()
    image = convert_from_rgb(image)
    return image


def load_image(img_path):
    image = np.asfarray(Image.open(img_path))  # from PIL image to numpy array
    return image


# Selects picture to be one of the screenshots, left center or right and adjusts angle
def randomise_picture(ss_paths, steering_angle):
    # selects randomly 0-left, 1-center, 2-right
    rnd = np.random.choice(3)
    img_path = ss_paths[rnd]
    image = load_image(img_path)

    if rnd == 0:
        steering_angle = steering_angle + 0.2
    elif rnd == 2:
        steering_angle = steering_angle - 0.2

    return image, steering_angle

# randomly mirror the image and the steering angle
def random_flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = np.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


# preprocess and augument dataset
def augument(ss_paths, steering_angle):
    image, steering_angle = randomise_picture(ss_paths, steering_angle)
    image = preprocess(image)  # apply the preprocessing    
    image, steering_angle = random_flip(image, steering_angle)

    # todo maybe add translate, shadow, brightness 
    return image, steering_angle


def load_csv_data(learning_set_path, validation_set_size):
    data_df = pd.read_csv(os.path.join(os.getcwd(), learning_set_path, 'driving_log.csv'),
                          names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    # yay dataframes, we can select rows and columns by their names
    # we'll store the camera images as our input data
    x = data_df[["left", "center", "right"]].values
    # and our steering commands as our output data
    y = data_df['steering'].values

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=validation_set_size, random_state=0, shuffle=True)

    return (x_train, y_train), (x_valid, y_valid)


def load_batch(ss_paths_set, steering_angles, idx, batch_size):
    '''
    input is paths to 3 images and, steering angle
    output is selected and altered image and adjusted steering_angle
    '''
    input_images_batch = []
    steering_angles_batch = []

    # load images for current iteration
    for ss_paths, steering_angle in zip(ss_paths_set[idx*batch_size:(idx+1)*batch_size],
                                         steering_angles[idx*batch_size:(idx+1)*batch_size]):
        image, steering_angle = augument(ss_paths, steering_angle)
        input_images_batch.append(image)
        steering_angles_batch.append(steering_angle)

    return input_images_batch, steering_angles_batch
