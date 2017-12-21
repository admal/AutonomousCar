import tflearn

from model2.Model import Model

from config import MODEL_DIRECTORY


class TrainModel(Model):
    def train(self, x, y, x_val, y_val):
        model = self.get_model()

        model.fit( x,
                  y,
                  n_epoch=100,
                  validation_set=( x_val,  y_val),
                  shuffle=True,
                  show_metric=True,
                  batch_size=128,
                  snapshot_step=10,
                  snapshot_epoch=False,
                  run_id='NvidiaModel')
        model.save(self.model_dir)