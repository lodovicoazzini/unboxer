import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy.ndimage import label

from utils.image_similarity.intensity_based import euclidean_distance


def distance_matrix(heatmaps, dist_func=euclidean_distance, show_map=False, names=None, annot=True, zero_diag=True):
    """
    Compute the distance matrix for a list of heatmaps
    """
    # initialize the distances matrix to 0
    num_heatmaps = len(heatmaps)
    dist_matrix = np.zeros(shape=(num_heatmaps, num_heatmaps))
    # compute the heatmaps distances above the diagonal
    for row in range(0, num_heatmaps - 1):
        for col in range(row + 1, num_heatmaps):
            lhs = heatmaps[row]
            rhs = heatmaps[col]
            dist_matrix[row][col] = dist_func(lhs, rhs)
    # complete the rest of the distance matrix by summing it to its transposed
    dist_matrix = dist_matrix + dist_matrix.T

    # visualize the heatmap
    if show_map and names is not None:
        dist_matrix_df = pd.DataFrame(
            dist_matrix,
            columns=names,
            index=names
        )
        dist_matrix_df = dist_matrix_df.groupby(dist_matrix_df.columns, axis=1).mean()
        dist_matrix_df = dist_matrix_df.groupby(dist_matrix_df.index, axis=0).mean()
        if zero_diag:
            np.fill_diagonal(dist_matrix_df.values, np.nan)
        fig = plt.figure(figsize=(10, 10))
        ax = sns.heatmap(
            dist_matrix_df,
            annot=annot,
            cmap='OrRd',
            linewidth=.1,
            vmin=0, vmax=1
        )
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        return dist_matrix_df, fig, ax

    return dist_matrix


def generate_contributions(
        explainer,
        data: np.ndarray,
        predictions: np.ndarray
):
    contributions = explainer.explain(data, predictions)
    # convert the contributions to grayscale
    if len(contributions.shape) > 3 and contributions.shape[-1] > 1:
        contributions = np.squeeze(tf.image.rgb_to_grayscale(contributions).numpy())
    # filter for the positive contributions
    contributions = np.ma.masked_less(np.squeeze(contributions), 0).filled(0)

    return contributions


def extract_maps_clusters():
    # Create the base dataframe with the features and the cells containing the clusters
    original_df = pd.read_pickle('in/featuremaps_data')
    original_df['mode'] = 'original'

    # Compute the merged clusters
    merged_df = pd.DataFrame.copy(original_df, deep=True)
    # Compute the number of items in each cell
    merged_df['cells_size'] = merged_df['clusters'].apply(
        np.vectorize(lambda cell: len(cell) if cell is not None else 0)
    )
    # Get the matrix for the connected components
    shape = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ]
    merged_df['connected_components'] = merged_df['cells_size'].apply(lambda matrix: label(matrix, shape)[0])
    # Get the list of merged clusters
    merged_df['clusters_list'] = merged_df.apply(
        lambda row: merge_clusters(row['clusters'], row['connected_components'])
        , axis=1
    )
    merged_df['mode'] = 'reduced'

    # Fix the columns before merging
    original_df['clusters_list'] = original_df['clusters'].apply(get_clusters_list)
    merged_df = merged_df.drop(columns=['connected_components', 'cells_size'])

    # Merge the two datasets
    complete_df = pd.concat([original_df, merged_df])
    complete_df = complete_df.drop(columns='clusters')
    complete_df = complete_df.rename({'clusters_list': 'clusters'}, axis=1)
    complete_df = complete_df.sort_values(['map_size', 'approach', 'mode']).reset_index(drop=True)

    complete_df.to_pickle('logs/feature_combinations_clusters')

    return complete_df


def get_clusters_list(matrix):
    clusters_flat = matrix.flatten()
    return clusters_flat[clusters_flat != None]


def merge_clusters(clusters_matrix, connected_components):
    merged_clusters = []
    for clusters_label in np.unique(connected_components):
        label_clusters = clusters_matrix[np.where(connected_components == clusters_label)]
        label_clusters = label_clusters[label_clusters != None]
        if len(label_clusters) > 0:
            merged_clusters.append(np.concatenate(label_clusters))

    return merged_clusters
