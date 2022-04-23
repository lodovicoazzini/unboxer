import numpy as np
from clusim.clustering import Clustering

from config_dirs import PREDICTIONS_PATH
from utils.dataset import get_train_test_data, get_data_masks


def get_frac_misses(cluster) -> float:
    """
    Find the fraction of misclassified elements in each cluster.
    """
    # Get the indexes for the misclassified elements
    # Get the indexes of the misclassified elements
    _, (test_data, test_labels) = get_train_test_data(rgb=True)
    predictions = np.loadtxt(PREDICTIONS_PATH)
    mask_miss, mask_label = get_data_masks(test_labels, predictions, label=5)
    mask_miss_label = mask_miss[mask_label]
    miss_idxs = np.argwhere(mask_miss_label)
    # Find the number of misclassified elements for each cluster
    return len([entry for entry in cluster if entry in miss_idxs]) / len(cluster)


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
