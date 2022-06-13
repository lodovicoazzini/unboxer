from itertools import combinations

import numpy as np
from clusim.clustering import Clustering


def __intra_pairs(clusters_membership: np.ndarray) -> set:
    # Find the labels for the clusters in the configuration
    clusters_labels = np.unique(clusters_membership)
    return set([
        pair
        # Iterate through the cluster labels
        for label in clusters_labels
        # intrapairs = combinations of two elements in the clusters
        for pair in combinations(np.where(clusters_membership == label)[0], 2)
    ])


def intra_pairs_similarity(lhs: Clustering, rhs: Clustering) -> float:
    """
    Compute the intrapairs similarity between two cluster configurations
    :param lhs: The first cluster configuration
    :param rhs: The second cluster configuration
    :return: The fraction of common intrapairs between the two configurations
    """
    # Convert the clusters to membership list
    lhs, rhs = lhs.to_membership_list(), rhs.to_membership_list()
    # Extract the intrapairs from the two configurations
    intra_lhs, intra_rhs = __intra_pairs(lhs), __intra_pairs(rhs)
    # Compute the fraction of common intrapairs
    score = len(intra_lhs.intersection(intra_rhs)) * 2 / (len(intra_lhs) + len(intra_rhs))
    return score
