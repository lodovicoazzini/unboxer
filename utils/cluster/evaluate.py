import numpy as np
import pandas as pd


def silhouette_score(dist_matrix, clusters):
    # compute the silhouette score for each point
    silhouette_scores = np.array([])
    for idx in range(0, dist_matrix.shape[0]):
        # compute the cohesion for the point (average intra-cluster distance)
        # find the label for the current point
        point_label = clusters[idx]
        # create a dataframe for the distances of the point
        point_distances = pd.DataFrame({
            'label': clusters,
            'distance': dist_matrix[1, :]
        }).reset_index()
        # filter for the points in the same clusters
        same_cluster_distances = point_distances[
            (point_distances['label'] == point_label) &
            (point_distances['index'] != idx)
            ]['distance'].values
        # compute the average intra distance, if empty (singleton) -> SI(i) = 1
        if point_label == -1 or len(same_cluster_distances) == 0:
            silhouette_scores = np.append(
                silhouette_scores,
                1
            )
            continue
        avg_intra_dist = np.average(same_cluster_distances) if len(same_cluster_distances) > 0 else 1
        # compute the separation for the point (minimum average inter-cluster distance)
        # initialize the minimum inter distance to infinite
        min_inter_dist = np.inf
        other_labels = set(clusters[clusters != point_label])
        for other_label in other_labels:
            # compute the average distance with the points of the cluster
            # compute the average distance with the points of the cluster
            other_cluster_distances = point_distances[point_distances['label'] == other_label]['distance'].values
            avg_other_dist = np.mean(other_cluster_distances)
            # update the value of the minimum distance
            min_inter_dist = avg_other_dist if avg_other_dist < min_inter_dist else min_inter_dist
        # append the silhouette score for the point to the list
        si_point = (min_inter_dist - avg_intra_dist) / max(avg_intra_dist, min_inter_dist)
        silhouette_scores = np.append(
            silhouette_scores,
            si_point
        )
    # compute the silhouette score for the clustering configuration (average silhouette for the points)
    return np.mean(silhouette_scores)
