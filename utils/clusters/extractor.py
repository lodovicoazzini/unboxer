from typing import Callable

import numpy as np

from utils import global_values
from utils.stats import compute_comparison_matrix


def get_misses_count(cluster: list) -> int:
    """
    Find the number of misclassified elements in a clusters
    :param cluster: The clusters
    :return: The number of misclassified elements in the clusters
    """
    # Get the indexes of the misclassified elements
    miss_idxs = np.argwhere(global_values.mask_miss_label).flatten()
    # Get the count of misclassified elements in the clusters
    return len([entry for entry in cluster if entry in miss_idxs])


def get_frac_misses(cluster: list) -> float:
    """
    Find the fraction of misclassified elements in a clusters
    :param cluster: The clusters
    :return: The fraction of misclassified elements in the clusters
    """
    # Get the count of misclassified elements in the clusters
    count_misses = get_misses_count(cluster)
    # Get the fraction of misclassified elements in the clusters
    return count_misses / len(cluster)


def get_labels_purity(cluster: list):
    # Get the indexes of the misclassified elements
    miss_idxs = np.argwhere(global_values.mask_miss_label).flatten()
    # Filter for the misclassified entries in the clusters
    masked_entries = [entry for entry in cluster if entry in miss_idxs]
    # No misclassified entries -> return np.nan
    if len(masked_entries) == 0:
        return np.nan
    # Find the predicted labels for the misclassified elements in the clusters
    masked_labels = global_values.predictions[global_values.mask_label][masked_entries]
    # Compute the purity of the clusters as the weighted average of the fraction of occurrences of each label
    labels, counts = np.unique(masked_labels, return_counts=True)
    purity = np.average(counts / len(masked_labels), weights=counts)
    return purity


def get_central_elements(
        cluster_idxs: list,
        cluster_elements: list,
        elements_count: int,
        metric: Callable,
) -> list:
    """
    Get the centroid and the closest elements in the cluster
    """
    # Compute the distance matrix
    dist_matrix = compute_comparison_matrix(values=cluster_elements, metric=metric)
    # Find the centroid of the cluster as the element with the least sum of the distances from the others
    cluster_idxs = np.arange(len(global_values.mask_label))[cluster_idxs]
    medoid_idx = np.argmin(np.apply_along_axis(np.nansum, axis=0, arr=dist_matrix))
    medoid = cluster_idxs[medoid_idx]
    # Set the image to itself to inf
    medoid_distances = np.nan_to_num(dist_matrix[medoid_idx], nan=np.inf)
    # Find the three closest elements
    closest = cluster_idxs[np.argsort(medoid_distances)][:elements_count - 1]
    print(medoid_distances)
    # Add the medoid
    central = list(np.insert(closest, 0, values=medoid))
    return central
