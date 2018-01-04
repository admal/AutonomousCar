#to save our model periodically as checkpoints for loading later
from keras.callbacks import ModelCheckpoint
#popular optimization strategy that uses gradient descent 
from keras.optimizers import Adam

from model3.Model import Model
from config import BATCH_SIZE, LEARNING_RATE, EPOCHS_COUNT, BATCH_SIZE, SAVE_BEST_ONLY
from utils import generate_batches


class TrainModel(Model):

    def __init__(self, model_dir=None):
        super(TrainModel, self).__init__(model_dir)

    # takes the models and performs a fit on given extracted batches
    def train(self, x_train, x_valid, y_train, y_valid):

        model = self.get_model()

        checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                    monitor='val_loss',
                                    verbose=0,
                                    save_best_only=SAVE_BEST_ONLY,
                                    mode='auto')

        model.compile(loss='mean_squared_error', optimizer=Adam(lr=LEARNING_RATE))

        # maybe batches here need the same size not sure
        batch_generator = None
        validation_batch = None
        # maybe batch - could be a delegate which would generate a random input sample
        # I think it is a delegate and it is run async for each epoch

        # this may iterate all the epochs at once?
        batches_count_for_each_epoch = int(len(x_train)/BATCH_SIZE)
        batches_count_for_validation = int(len(x_valid)/BATCH_SIZE)
        model.fit_generator(generator=generate_batches(x_train, y_train, augument_pictures=True),
                            steps_per_epoch=batches_count_for_each_epoch,
                            epochs=EPOCHS_COUNT,
                            max_q_size=1,
                            validation_data=generate_batches(x_valid, y_valid, augument_pictures=False),
                            nb_val_samples=batches_count_for_validation,
                            callbacks=[checkpoint],
                            verbose=1)
