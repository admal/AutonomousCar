#The Sequential container is a linear stack of layers
from keras.models import Sequential, load_model
#what types of layers do we want our model to have?
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from config import INPUT_SHAPE, KEEP_PROB


class Model:
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)
    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
    """
    def __init__(self, model_dir=None):
        super(Model, self).__init__()
        self._model = None
        self._model_dir = model_dir


    def build_model(self):
        self._model = Sequential()
        self._model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
        self._model.add(Conv2D(24, 5, 5, activation="elu", subsample=(2, 2)))
        self._model.add(Conv2D(36, 5, 5, activation="elu", subsample=(2, 2)))
        self._model.add(Conv2D(48, 5, 5, activation="elu", subsample=(2, 2)))
        self._model.add(Conv2D(64, 3, 3, activation="elu"))
        self._model.add(Conv2D(64, 3, 3, activation="elu"))
        self._model.add(Dropout(KEEP_PROB))
        self._model.add(Flatten())
        self._model.add(Dense(100, activation="elu"))
        self._model.add(Dense(50, activation="elu"))
        self._model.add(Dense(10, activation="elu"))
        self._model.add(Dense(1))
        self._model.summary()

        return self._model


    def get_model(self):
        if self._model is None:
            if self._model_dir is None:
                self.build_model()
            else:
                self._model = load_model(self._model_dir)

        return self._model






