from model.Model import Model
import tensorflow as tf
import numpy as np
from log import *


class TrainModel(Model):
    def train(self, x_train, y_train):
        self._model = tf.estimator.Estimator(model_fn=self.build_model, model_dir=self.model_dir)

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.asarray(x_train, dtype=np.float16)},
            y=np.asarray(y_train),
            num_epochs=10,
            shuffle=False
        )
        log_info("Start training")
        self._model.train(train_input_fn, steps=1)
        self.isTrained = True
        log_info("End training")

    def evaluate(self, x_val, y_val):
        if self.isTrained:
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": np.asarray(x_val, dtype=np.float16)},
                y=np.array(y_val, dtype=np.float16),
                num_epochs=None,
                shuffle=False)
            log_info("Start evaluating")
            results = self._model.evaluate(input_fn=train_input_fn)
            self._model.export_savedmodel()
            print("Loss %s" % results["loss"])
            print("Root Mean Squared Error: %s" % results["rmse"])