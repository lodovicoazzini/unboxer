import numpy as np
from sklearn.cluster import DBSCAN

from utils.cluster.evaluate import silhouette_score


def density_based_cluster(dist_matrix, min_samples=5, eps=.5, verbose=False):
    """
    Clusters the heatmaps based on a distance matrix
    :param dist_matrix: The distance matrix to use for the clusters
    :param min_samples: The minimum number of points to make a cluster
    :param eps: The distance for two points to be neighbors
    :param verbose: Print information about the clusters
    :return: The clusters' labels
    """
    # generate the clusters
    clusters = DBSCAN(metric='precomputed', min_samples=min_samples, eps=eps).fit_predict(dist_matrix)

    if verbose:
        print(f"""
Silhouette score = {silhouette_score(dist_matrix, clusters)}
    min_samples = {min_samples}
    eps = {eps} 
        """)

    return clusters


def density_based_cluster_tuned(dist_matrix, min_samples_list, eps=.5, verbose=False):
    """
    Clusters the heatmaps based on a distance matrix
    :param dist_matrix: The distance matrix to use for the clusters
    :param min_samples_list: The list of values to try for the minimum number of points in a cluster
    :param eps: The distance for two points to be neighbors
    :param verbose: Print information about the clusters
    :return: The clusters' labels
    """
    # compute the silhouette scores for the clustering configurations
    silhouette_scores = np.array([])
    for min_samples in min_samples_list:
        config_clusters = density_based_cluster(dist_matrix, min_samples=min_samples, eps=eps)
        silhouette_scores = np.append(silhouette_scores, silhouette_score(dist_matrix, config_clusters))

    # get value corresponding to the minimum silhouette score
    min_samples = min_samples_list[np.argmax(silhouette_scores)]

    return density_based_cluster(dist_matrix, min_samples=min_samples, eps=eps, verbose=verbose)
