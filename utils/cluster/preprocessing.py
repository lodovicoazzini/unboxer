import numpy as np
import tensorflow as tf

from utils.image_similarity.intensity_based import euclidean_distance


def distance_matrix(heatmaps, dist_func=euclidean_distance):
    """
    Compute the distance matrix for a list of heatmaps
    :param heatmaps: The list of heatmaps
    :param dist_func: The distance function to use (default is Euclidean distance)
    :return: The distance matrix NxN for the N heatmaps
    """
    # initialize the distances matrix to 0
    num_heatmaps = len(heatmaps)
    dist_matrix = np.zeros(shape=(num_heatmaps, num_heatmaps))
    # compute the heatmaps distances above the diagonal
    for row in range(0, num_heatmaps - 1):
        for col in range(row + 1, num_heatmaps):
            lhs = heatmaps[row]
            rhs = heatmaps[col]
            # convert te contributions to grayscale if necessary
            lhs = tf.image.rgb_to_grayscale(lhs) if len(lhs.shape) > 2 and lhs.shape[-1] > 1 else lhs
            rhs = tf.image.rgb_to_grayscale(rhs) if len(rhs.shape) > 2 and rhs.shape[-1] > 1 else rhs
            # reduce to a 2D array
            lhs = np.squeeze(lhs)
            rhs = np.squeeze(rhs)
            dist_matrix[row][col] = dist_func(lhs, rhs)
    # complete the rest of the distance matrix by summing it to its transposed
    dist_matrix = dist_matrix + dist_matrix.T

    return dist_matrix