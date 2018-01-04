import numpy as np

from model3.Model import Model
from config import MODEL_DIRECTORY


class PredictModel(Model):

    def __init__(self, model_dir):
        super(PredictModel, self).__init__(model_dir)


    def predict(self, image):
        model = self.get_model()

        #put in array
        image = np.array([image])
        steering_angle = model.predict(image, batch_size=1)

        return steering_angle
