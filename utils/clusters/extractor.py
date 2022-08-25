from typing import Callable

import numpy as np

from utils import global_values
from utils.clusters.postprocessing import get_misclassified_items
from utils.stats import compute_comparison_matrix


def get_misses_count(cluster: list) -> int:
    """
    Find the number of misclassified elements in a clusters
    :param cluster: The clusters
    :return: The number of misclassified elements in the clusters
    """
    mask_label = np.array(global_values.test_labels == global_values.EXPECTED_LABEL)
    mask_miss = np.array(global_values.test_labels != global_values.predictions)
    # Get the indexes of the misclassified elements
    miss_idxs = np.argwhere(mask_miss[mask_label]).flatten()
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
    # Filter for the misclassified entries in the clusters
    cluster = np.array(cluster)
    masked_entries = get_misclassified_items(cluster)
    # No misclassified entries -> return np.nan
    if len(masked_entries) == 0:
        return np.nan
    # Find the predicted labels for the misclassified elements in the clusters
    masked_labels = global_values.predictions[global_values.test_labels == global_values.EXPECTED_LABEL][masked_entries]
    # Compute the purity of the clusters as the weighted average of the fraction of occurrences of each label
    labels, counts = np.unique(masked_labels, return_counts=True)
    purity = np.average(counts / len(masked_labels), weights=counts)
    return purity


def get_central_elements(
        cluster_idxs: list,
        cluster_elements: list,
        elements_count: int,
        metric: Callable,
        show_progress_bar: bool = False
) -> list:
    """
    Get the centroid and the closest elements in the cluster
    """
    # Compute the distance matrix
    dist_matrix = compute_comparison_matrix(
        values=cluster_elements,
        metric=metric,
        show_progress_bar=show_progress_bar,
        multi_process=False
    )
    # Find the centroid of the cluster as the element with the least sum of the distances from the others

    mask_label = np.array(global_values.test_labels == global_values.EXPECTED_LABEL)
    cluster_idxs = np.arange(len(mask_label))[cluster_idxs]
    medoid_idx = np.argmin(np.apply_along_axis(np.nansum, axis=0, arr=dist_matrix))
    medoid = cluster_idxs[medoid_idx]
    # Set the image to itself to inf
    medoid_distances = dist_matrix[medoid_idx]
    # Find the three closest elements
    closest = cluster_idxs[np.argsort(medoid_distances)][:elements_count]
    # Add the medoid
    return closest
