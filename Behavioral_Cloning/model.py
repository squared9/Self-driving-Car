import os
import errno
import json
import batcher
import pandas
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D
from keras.layers import Flatten, Activation, Dense, MaxPooling2D, Dropout

# model and training constants
LEARNING_RATE = 0.0005
EPOCHS = 20
NUMBER_OF_IMAGES = 24064  ## make them divisible by 128 instead of total of 24108
TRAIN_BATCH_SIZE = 19200
VALIDATION_BATCH_SIZE = 4864
BATCH_AMOUNT = 128

# files
MODEL_FILENAME = "model.json"
MODEL_WEIGHTS_FILENAME = "model.h5"
DATA_PATH = "data/"
DRIVING_LOG_FILENAME = DATA_PATH + "driving_log.csv"


def create_model():
    """
    Creates a Keras model based on NVidia's DAVE-2 model as described in:
    http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    Uses max pooling in convolutional layers and dropouts in dense layers to minimize overfitting
    :return: Keras model
    """
    # NVidia DAVE-2
    dave2 = Sequential()

    # 200x66x3 -> 98x31x24, 5x5 convolution
    dave2.add(Convolution2D(24, 5, 5, border_mode="valid", input_shape=(66, 200, 3), dim_ordering="tf"))
    dave2.add(Activation("relu"))
    dave2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="same"))

    # 98x31x24 -> 47x14x36, 5x5 convolution
    dave2.add(Convolution2D(36, 5, 5, border_mode="valid"))
    dave2.add(Activation("relu"))
    dave2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="same"))

    # 47x14x36 -> 22x5x48, 5x5 convolution
    dave2.add(Convolution2D(48, 5, 5, border_mode="valid"))
    dave2.add(Activation("relu"))
    dave2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode="same"))

    # 22x5x48 -> 20x3x64, 3x3 convolution
    dave2.add(Convolution2D(64, 3, 3, border_mode="valid"))
    dave2.add(Activation("relu"))
    dave2.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), border_mode="same"))

    # 20x3x64 -> 18x1x64, 3x3 convolution
    dave2.add(Convolution2D(64, 3, 3, border_mode="valid"))
    dave2.add(Activation("relu"))
    dave2.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), border_mode="same"))

    # flatten
    dave2.add(Flatten())

    # dense 1164
    dave2.add(Dense(1164))
    dave2.add(Activation("relu"))
    dave2.add(Dropout(0.5))

    # dense 100
    dave2.add(Dense(100))
    dave2.add(Activation("relu"))

    # dense 50
    dave2.add(Dense(50))

    # dense 10
    dave2.add(Dense(10))

    # dense 1
    dave2.add(Dense(1))

    dave2.compile(loss="mse", optimizer=Adam(LEARNING_RATE))

    return dave2


def train_model(model):
    """
    Trains model from a dataset
    :param model:
    :return:
    """
    # read driving log first
    driving_log = pandas.read_csv(DRIVING_LOG_FILENAME)

    # now training
    training_batch = batcher.next(driving_log, BATCH_AMOUNT, DATA_PATH)
    validation_batch = batcher.next(driving_log, BATCH_AMOUNT, DATA_PATH)

    training = model.fit_generator(training_batch, verbose=1, nb_epoch=EPOCHS, samples_per_epoch=TRAIN_BATCH_SIZE,
                              validation_data=validation_batch, nb_val_samples=VALIDATION_BATCH_SIZE)


def save(model, model_filename, model_weights_filename):
    """
    Saves model and its weights
    :param model:
    :param model_filename:
    :param model_weights_filename:
    :return:
    """
    # save model
    try:
        os.remove(model_filename)
    except OSError as error:
        if error.errno != errno.ENOENT:
            raise
    content = model.to_json()
    with open(model_filename, 'w') as file:
        json.dump(content, file)

    # save weights
    try:
        os.remove(model_weights_filename)
    except OSError as error:
        if error.errno != errno.ENOENT:
            raise
    model.save_weights(model_weights_filename)


dave2 = create_model()
dave2.summary()
train_model(dave2)
save(dave2, MODEL_FILENAME, MODEL_WEIGHTS_FILENAME)