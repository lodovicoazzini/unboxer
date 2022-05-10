from collections import Counter

import numpy as np
from clusim.clustering import Clustering

from config.config_dirs import PREDICTIONS_PATH
from utils.dataset import get_train_test_data, get_data_masks


def get_misses_count(cluster, predictions=None) -> float:
    """
    Find the number of misclassified elements in a cluster.
    """
    # Get the indexes for the misclassified elements
    # Get the indexes of the misclassified elements
    if predictions is None:
        predictions = np.loadtxt(PREDICTIONS_PATH)
    _, (test_data, test_labels) = get_train_test_data(rgb=True)
    mask_miss, mask_label = get_data_masks(test_labels, predictions, label=5)
    mask_miss_label = mask_miss[mask_label]
    miss_idxs = np.argwhere(mask_miss_label).flatten()
    # Find the number of misclassified elements for each cluster
    return len([entry for entry in cluster if entry in miss_idxs])


def get_frac_misses(cluster, predictions=None) -> float:
    """
    Find the fraction of misclassified elements in each cluster.
    """
    # Get the indexes for the misclassified elements
    # Get the indexes of the misclassified elements
    count_misses = get_misses_count(cluster, predictions=predictions)
    # Find the number of misclassified elements for each cluster
    return count_misses / len(cluster)


def get_labels_purity(cluster, predictions=None) -> float:
    _, (test_data, test_labels) = get_train_test_data(rgb=True)
    if predictions is None:
        predictions = np.loadtxt(PREDICTIONS_PATH)

    mask_miss, mask_label = get_data_masks(test_labels, predictions, label=5)
    mask_miss_label = mask_miss[mask_label]
    miss_idxs = np.argwhere(mask_miss_label).flatten()
    # Find the misclassified entries in the cluster
    masked_entries = [entry for entry in cluster if entry in miss_idxs]
    # If there are no misclassified entries return -1
    if len(masked_entries) == 0:
        return 0

    # Find the labels for the misclassified elements in the cluster
    masked_labels = predictions[mask_label][masked_entries]
    # Compute the purity of the cluster as the weighted average of the fraction of occurrences of each label
    labels, counts = np.unique(masked_labels, return_counts=True)
    purity = np.average(counts / len(masked_labels), weights=counts)

    return purity


def sorted_clusters(clusters, metric: callable):
    """
    Sort a list of clusters based on a metric
    """
    sorting_list = [metric(cluster) for cluster in clusters]
    zipped = list(zip(clusters, sorting_list))
    sorted_zipped = sorted(zipped, key=lambda entry: entry[1])
    return [cluster for cluster, _ in sorted_zipped]


def get_non_unique_membership_list(clusters) -> list:
    """
    Get the membership list for a list of overlapping clusters.
    In the case of overlap the first occurrence is returned.
    """
    return [
        list(clus_id)[0]
        for idx, clus_id
        in sorted(Clustering().from_cluster_list(clusters).to_elm2clu_dict().items())
    ]


def get_filtered_clusters(clusters, mask):
    """
    Get the clusters for the masked elements.
    """
    # Get the indexes for the mask
    masked_idxs = np.argwhere(mask).flatten()
    # Filter the elements in the clusters for the misclassified ones
    membership = np.array(Clustering().from_cluster_list(clusters).to_membership_list())
    masked_membership = membership[masked_idxs]
    masked_clusters = Clustering().from_membership_list(masked_membership).to_cluster_list()

    return masked_clusters


def get_clusters_containing(clusters, mask):
    """
    Get the clusters containing at least one masked element.
    """
    # Get the indexes for the mask
    masked_idxs = np.argwhere(mask).flatten()
    # Filter the clusters for those containing at least one masked element
    masked_clusters = [cluster for cluster in clusters if len(set(cluster).intersection(set(masked_idxs))) > 0]

    return masked_clusters


def get_popularity_score(clusters_config, all_clusters, sample_size) -> float:
    clusters_popularity = Counter(map(tuple, all_clusters))
    return np.average([clusters_popularity[tuple(cluster)] / sample_size for cluster in clusters_config])
