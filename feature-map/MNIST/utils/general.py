import logging as log
import sys

# For Python 3.6 we use the base keras
import keras
import matplotlib.pyplot as plt
import pandas as pd

# from tensorflow import keras

# local imports

IMG_SIZE = 28

import numpy as np


def compute_sparseness(map, x):
    n = len(map)
    # Sparseness is evaluated only if the archive is not empty
    # Otherwise the sparseness is 1
    if (n == 0) or (n == 1):
        sparseness = 0
    else:
        sparseness = density(map, x)
    return sparseness


def get_neighbors(b):
    neighbors = []
    neighbors.append((b[0], b[1] + 1))
    neighbors.append((b[0] + 1, b[1] + 1))
    neighbors.append((b[0] - 1, b[1] + 1))
    neighbors.append((b[0] + 1, b[1]))
    neighbors.append((b[0] + 1, b[1] - 1))
    neighbors.append((b[0] - 1, b[1]))
    neighbors.append((b[0] - 1, b[1] - 1))
    neighbors.append((b[0], b[1] - 1))

    return neighbors


def density(map, x):
    b = x.features
    density = 0
    neighbors = get_neighbors(b)
    for neighbor in neighbors:
        if neighbor not in map:
            density += 1
    return density


def input_reshape(x):
    # shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        x_reshape = x.reshape(x.shape[0], 1, 28, 28)
    else:
        x_reshape = x.reshape(x.shape[0], 28, 28, 1)
    x_reshape = x_reshape.astype('float32')
    x_reshape /= 255.0

    return x_reshape


def get_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def print_image(filename, image, cmap=''):
    if cmap != '':
        plt.imsave(filename, image.reshape(28, 28), cmap=cmap, format='png')
    else:
        plt.imsave(filename, image.reshape(28, 28), format='png')
    np.save(filename, image)


# Useful function that shapes the input in the format accepted by the ML model.
def reshape(v):
    v = (np.expand_dims(v, 0))
    # Shape numpy vectors
    if keras.backend.image_data_format() == 'channels_first':
        v = v.reshape(v.shape[0], 1, IMG_SIZE, IMG_SIZE)
    else:
        v = v.reshape(v.shape[0], IMG_SIZE, IMG_SIZE, 1)
    v = v.astype('float32')
    v = v / 255.0
    return v


def setup_logging(log_to, debug):
    def log_exception(extype, value, trace):
        log.exception('Uncaught exception:', exc_info=(extype, value, trace))

    # Disable annoyng messages from matplot lib.
    # See: https://stackoverflow.com/questions/56618739/matplotlib-throws-warning-message-because-of-findfont-python
    log.getLogger('matplotlib.font_manager').disabled = True

    term_handler = log.StreamHandler()
    log_handlers = [term_handler]
    start_msg = "Started test generation"

    if log_to is not None:
        file_handler = log.FileHandler(log_to, 'a', 'utf-8')
        log_handlers.append(file_handler)
        start_msg += " ".join(["writing to file: ", str(log_to)])

    log_level = log.DEBUG if debug else log.INFO

    log.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=log_level, handlers=log_handlers)

    sys.excepthook = log_exception

    log.info(start_msg)


def missing(s: pd.Series):
    """
    :return: The number of NaN values in a series
    """
    return s.isnull().sum()
