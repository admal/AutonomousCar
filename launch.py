# parsing command line arguments
import argparse
# decoding camera images
import base64
# for frametimestamp saving
from datetime import datetime
# high level file operations
import shutil
# real-time server
import socketio
# web server gateway interface
import eventlet.wsgi
# web framework
from flask import Flask
# input output
from io import BytesIO
from model2.PredictModel import PredictModel
from utils import *
from config import *

# initialize our server
sio = socketio.Server()
# our flask (web) app
app = Flask(__name__)
# init our model and image array as empty
model = PredictModel(TRAINED_MODEL_DIRECTORY)

model.load()
prev_image_array = None

# and a speed limit
speed_limit = MAX_SPEED


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
                steering_angle = result[0][0]

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
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
