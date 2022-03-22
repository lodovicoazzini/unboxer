import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN

from utils.cluster.evaluate import silhouette_score


def density_based_cluster(dist_matrix, min_samples=5, eps=.5, plot=False, verbose=False):
    """
    Clusters the heatmaps based on a distance matrix
    :param dist_matrix: The distance matrix to use for the clusters
    :param min_samples: The minimum number of points to make a cluster
    :param eps: The distance for two points to be neighbors
    :param plot: Whether to visualize the clusters
    :param verbose: Print information about the clusters
    :return: The clusters' labels (+ fig, ax if plot == True)
    """
    # generate the clusters
    clusters = DBSCAN(metric='precomputed', min_samples=min_samples, eps=eps).fit_predict(dist_matrix)

    if verbose:
        print(f"""
Silhouette score = {silhouette_score(dist_matrix, clusters)}
    min_samples = {min_samples}
    eps = {eps} 
        """)
    if plot:
        # prepare the general figure
        fig, ax = plt.subplots(figsize=(16, 9))
        sns.despine()

        # associate one cluster label for each color
        colors = [plt.cm.Spectral(val) for val in np.linspace(0, 1, len(set(clusters)))]
        colors_dict = {
            label: colors[idx]
            for idx, label in enumerate(set(clusters))
        }
        # associate black with noise
        colors_dict[-1] = (0, 0, 0, 1)

        for label, color in colors_dict.items():
            # plot the core samples
            points = dist_matrix[clusters == label]
            ax.plot(
                points[:, 0], points[:, 1],
                'x' if label == -1 else 'o', markerfacecolor=tuple(color), markeredgecolor='k',
                label='no cluster' if label == -1 else label
            )
            plt.legend()
            ax.set_xlim((0, ax.get_xlim()[1]))
            ax.set_ylim((0, ax.get_ylim()[1]))

        return clusters, fig, ax

    return clusters


def density_based_cluster_tuned(dist_matrix, min_samples_list, eps=.5, plot=False, verbose=False):
    """
    Clusters the heatmaps based on a distance matrix
    :param dist_matrix: The distance matrix to use for the clusters
    :param min_samples_list: The list of values to try for the minimum number of points in a cluster
    :param eps: The distance for two points to be neighbors
    :param plot: Whether to visualize the clusters
    :param verbose: Print information about the clusters
    :return: The clusters' labels (+ fig, ax if plot == True)
    """
    # compute the silhouette scores for the clustering configurations
    silhouette_scores = np.array([])
    for min_samples in min_samples_list:
        config_clusters = density_based_cluster(dist_matrix, min_samples=min_samples, eps=eps)
        silhouette_scores = np.append(silhouette_scores, silhouette_score(dist_matrix, config_clusters))

    # get value corresponding to the minimum silhouette score
    min_samples = min_samples_list[np.argmax(silhouette_scores)]

    return density_based_cluster(dist_matrix, min_samples=min_samples, eps=eps, plot=plot, verbose=verbose)
