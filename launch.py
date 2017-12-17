# parsing command line arguments
import argparse
# decoding camera images
import base64
# for frametimestamp saving
from datetime import datetime
# reading and writing files
import os
# high level file operations
import shutil
# matrix math
import numpy as np
# real-time server
import socketio
# concurrent networking
import eventlet
# web server gateway interface
import eventlet.wsgi
# image manipulation
from PIL import Image
# web framework
from flask import Flask
# input output
from io import BytesIO

from model.PredictModel import PredictModel
from utils import *

# initialize our server
sio = socketio.Server()
# our flask (web) app
app = Flask(__name__)
# init our model and image array as empty
model = PredictModel("C:\\Studies\\AI\\Model")
model.load()
prev_image_array = None

MAX_SPEED, MIN_SPEED = 25, 10
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3

# and a speed limit
speed_limit = MAX_SPEED
BATCH_SIZE = 20
EPOCH_COUNT = 5


def load_nn_model(model_path):
    raise NotImplementedError("model loading should be implemented.")

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        print('received:: speed:{}'.format(speed))
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asfarray(image)  # from PIL image to numpy array
            image = preprocess(image)  # apply the preprocessing
            image = np.array([image])  # the model expects 4D array ??

            # predict the steering angle for the image
            # steering_angle = float(model.predict(image, batch_size=1))
            if model is not None:
                result = model.predict(image)
                steering_angle = result[0]

            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2

            print('sending:: steering angle:{} throttle:{}'.format(steering_angle, throttle))
            print()
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        nargs='?',
        default=None,
        help='Path to model'
    )
    parser.add_argument(
        'training_set',
        type=str,
        nargs='?',
        default=None,
        help='Path to learning set.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # learning set
    if args.training_set:
        print("LEARNING")
        validation_set_size = 0.1  # percentage of frames devoted to validation (last frames of the recording)
        training_csv_set, validation_csv_set = load_csv_data(args.training_set, validation_set_size)

    # load model
    if not args.training_set and args.model:
        print("LOADING MODEL")
        model = load_nn_model(args.model)


    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
