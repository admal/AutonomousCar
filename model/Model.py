import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from config import *

tf.logging.set_verbosity(tf.logging.ERROR)


class Model:
    isTrained = False
    _model = None
    _training_data = None
    image_width = IMAGE_WIDTH
    image_height = IMAGE_HEIGHT
    color_channels = IMAGE_CHANNELS
    model_dir = "C:\\Users\\ASUS\\Documents\\PW\\SieciNeuronowe\\Projekt2\\Model"

    def __init__(self, model_dir = None):
        """
        Initializes model
        :param model_dir: directory with model, it can be trained and used to predict rotation
        or with checkpoints to continue training
        """
        if model_dir is None:
            "C:\\Users\\ASUS\\Documents\\PW\\SieciNeuronowe\\Projekt2\\Model"
        else:
            self.model_dir = model_dir


    def build_model(self, features, labels, mode):

        conv1 = tf.layers.conv2d(
            inputs=features["x"],
            filters=24,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu
        )
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=36,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu
        )
        pool1 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=2,
            strides=2,
            padding="same"
        )
        conv3 = tf.layers.conv2d(
            inputs=pool1,
            filters=48,
            kernel_size=[3,3],
            padding="same",
            activation=tf.nn.relu
        )
        conv4 = tf.layers.conv2d(
            inputs=conv3,
            filters=64,
            kernel_size=[3,3],
            padding="same",
            activation=tf.nn.relu
        )
        pool2 = tf.layers.max_pooling2d(
            inputs=conv4,
            pool_size=2,
            strides=2,
            padding="same"
        )
        flatten = tf.layers.flatten(
            inputs=pool2
        )
        dense1 = tf.layers.dense(
            inputs=flatten,
            units=100,
            activation=tf.nn.relu
        )
        dense2 = tf.layers.dense(
            inputs=dense1,
            units=50,
            activation=tf.nn.relu
        )
        dense3 = tf.layers.dense(
            inputs=dense2,
            units=10,
            activation=tf.nn.relu
        )
        #maybe add dropout
        output = tf.layers.dense(
            inputs=dense3,
            units=1
        )

        predictions = tf.reshape(output, [-1])

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"rotations": predictions})

        loss = tf.losses.mean_squared_error(labels, predictions)

        # Calculate root mean squared error as additional eval metric
        eval_metric_ops = {
            "rmse": tf.metrics.root_mean_squared_error(
                tf.cast(labels, tf.float16), predictions)
        }

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops)


    def load(self):
        """
        Loads trained model using `self.model_dir`
        :return:
        """
        self._model = tf.estimator.Estimator(model_fn=self.build_model, model_dir=self.model_dir)
        self.isTrained = True