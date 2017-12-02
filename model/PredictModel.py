from model.Model import Model
import tensorflow as tf
import numpy as np


class PredictModel(Model):

    def predict(self, image):
        """
        Method uses trained model to predict rotation.
        :param image: array of pixels
        :returns single element array with 1/rotation
        """
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.asarray(image, dtype=np.float16)},
            num_epochs=1,
            shuffle=False
        )
        predictions = self._model.predict(input_fn=input_fn)
        result = []
        for i, p in enumerate(predictions):
            print("Prediction %s: %s" % (i + 1, p["rotations"]))
            result.append(p["rotations"])
        return result
