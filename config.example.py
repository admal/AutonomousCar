#LOGGING
IS_VERBOSE = True
TRAIN_LOG_FILE = "training.log"

#IMAGE PROCESSING
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3

#MODEL
LEARNING_RATE = 0.1
BATCH_SIZE = 20
MAX_ITERS = 1
EPOCHS_COUNT = 10
TRAIN_STEP_COUNT = 20
DATA_CSV_FILE="/home/auser/TrainingData"
#directory to save model during training
MODEL_DIRECTORY="/home/auser/Model"
#directory with already trained model, used during car simulation
TRAINED_MODEL_DIRECTORY = "/home/auser/TrainedModel"
EPOCHS_COUNT=100

#MISCELLANEOUS
MAX_SPEED = 25
MIN_SPEED = 10

#KERAS SETTINGS
KEEP_PROB = 0.5
SAVE_BEST_ONLY = True
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
