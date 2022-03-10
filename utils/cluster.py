import numpy as np
import tensorflow as tf


def heatmap_distance(lhs, rhs):
    # convert te contributions to grayscale if necessary
    lhs = tf.image.rgb_to_grayscale(lhs) if len(lhs.shape) > 2 and lhs.shape[-1] > 1 else lhs
    rhs = tf.image.rgb_to_grayscale(rhs) if len(rhs.shape) > 2 and rhs.shape[-1] > 1 else rhs
    # reduce to a 2D array
    lhs = np.squeeze(lhs)
    rhs = np.squeeze(rhs)

    # compute and return the distance between teh two matrices
    return np.sqrt(np.sum((lhs - rhs) ** 2))
