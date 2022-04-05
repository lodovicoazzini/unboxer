from itertools import combinations

import numpy as np


def intra_pairs(clusters: np.ndarray) -> set[tuple]:
    # find the labels for the clusters in the solution
    labels = np.unique(clusters)
    return set([
        pair
        # iterate through the labels
        for label in labels
        # intrapairs = combinations of two elements in the cluster
        for pair in combinations(np.where(clusters == label)[0], 2)
    ])


def intra_pairs_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    # compute the intra-pairs in the two solutions
    intra_lhs, intra_rhs = intra_pairs(lhs), intra_pairs(rhs)
    # return the fraction of common intrapairs
    return len(intra_lhs.intersection(intra_rhs)) * 2 / (len(intra_lhs) + len(intra_rhs))
