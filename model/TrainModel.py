import tensorflow as tf
import numpy as np
from log import *

from model.Model import Model
from config import *


class TrainModel(Model):
    def train(self, x_train, y_train):
        self._model = tf.estimator.Estimator(model_fn=self.build_model, model_dir=self.model_dir)

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.asarray(x_train, dtype=np.float16)},
            y=np.asarray(y_train),
            num_epochs=None,
            shuffle=True
        )
        log_info("Start training")
        self._model.train(train_input_fn, steps=TRAIN_STEP_COUNT)
        self.isTrained = True
        log_info("End training")

    def evaluate(self, x_val, y_val):
        if self.isTrained:
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": np.asarray(x_val, dtype=np.float16)},
                y=np.array(y_val, dtype=np.float16),
                num_epochs=1,
                shuffle=True)
            results = self._model.evaluate(input_fn=train_input_fn)
            log_info("Loss %s" % results["loss"])
            log_info("Root Mean Squared Error: %s" % results["rmse"])
