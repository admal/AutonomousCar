import csv
import collections
from tensorflow.python.platform import gfile
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import matplotlib.image as mpimg

tf.logging.set_verbosity(tf.logging.DEBUG)

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
def load_csv_without_header(filename,
                            target_dtype,
                            features_dtype,
                            target_column=-1,
                            data_column=2):
  """Load dataset from CSV file without a header row."""
  with gfile.Open(filename) as csv_file:
    data_file = csv.reader(csv_file)
    data, target = [], []
    for row in data_file:
      target.append(row.pop(target_column))
      # data.append(np.asarray(row, dtype=features_dtype))
      data.append(row.pop(data_column))

  target = np.array(target, dtype=target_dtype)
  data = np.array(data)
  return Dataset(data=data, target=target)


def load_image(image_file):
    return mpimg.imread(image_file)


class Model:
    isTrained = 0
    _model = None
    _training_data = None

    def __init__(self, trainingpath):
        self._training_data = load_csv_without_header(
            filename=trainingpath,
            target_dtype=np.float16,
            features_dtype=np.float16,
            target_column=3
        )

    image_width = 320
    image_height = 160
    color_channels = 3

    def build_model(self, features, labels, mode):
        print("start building model")
        # center_images = features["x"][2]
        # input_layer = tf.reshape(features, [-1, self.image_width, self.image_height, self.color_channels])
        # input_layer = tf.reshape(features["x"], [-1, 160, 320, self.color_channels])
        # input_layer = tf.reshape(features["x"], [-1, self.image_width, self.image_height, self.color_channels])
        #normalize data
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

        conv3 = tf.layers.conv2d(
            inputs=conv2,
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

        flatten = tf.layers.flatten(
            inputs=conv4
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

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops)

    def train(self):
        if self.isTrained == 0:
            self._model = tf.estimator.Estimator(model_fn=self.build_model)
            images = []

            for center_image in self._training_data.data:
                image = np.asarray(load_image(center_image))
                normalized_image = image/127.5 - 1.0

                images.append(normalized_image)

            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": np.asarray(images ,dtype=np.float16)},
                y=np.asarray(self._training_data.target,dtype=np.float16),
                num_epochs=None,
                shuffle=False)
            print("Start training")
            self._model.train(input_fn= train_input_fn, steps=10)
            print("Finised training")
            self.isTrained = 1

    def evaluate(self):
        if self.isTrained == 1:
            images = []
            for center_image in self._training_data.data:
                image = np.asarray(load_image(center_image))
                normalized_image = image/127.5 - 1.0

                images.append(normalized_image.flatten())

            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": np.asarray(images ,dtype=np.float16)},
                y=np.array(self._training_data.target,dtype=np.float16),
                num_epochs=None,
                shuffle=True)
            results = self._model.evaluate(input_fn=train_input_fn)
            print("Loss %s" % results["loss"])
            print("Root Mean Squared Error: %s" % results["rmse"])

    def predict(self):
        # Print out predictions
        # predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        #     x={"x": prediction_set.data},
        #     num_epochs=1,
        #     shuffle=False)
        predictions = self._model.predict(input_fn=None)
        for i, p in enumerate(predictions):
            print("Prediction %s: %s" % (i + 1, p["rotations"]))
    #
    # def run(self):
    #     if self.isTrained == 0:
    #         self._model = tf.estimator.Estimator(model_fn=self.build_model)
