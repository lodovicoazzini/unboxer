import numpy as np

from utils import global_values


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


def get_labels_purity(cluster: list) -> float:
    # Get the indexes of the misclassified elements
    miss_idxs = np.argwhere(global_values.mask_miss_label).flatten()
    # Filter for the misclassified entries in the clusters
    masked_entries = [entry for entry in cluster if entry in miss_idxs]
    # No misclassified entries -> return 0
    if len(masked_entries) == 0:
        return 0
    # Find the predicted labels for the misclassified elements in the clusters
    masked_labels = global_values.predictions[global_values.mask_label][masked_entries]
    # Compute the purity of the clusters as the weighted average of the fraction of occurrences of each label
    labels, counts = np.unique(masked_labels, return_counts=True)
    purity = np.average(counts / len(masked_labels), weights=counts)
    return purity
