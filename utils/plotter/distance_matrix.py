import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utils.general import show_progress


def show_distance_matrix(
        clusters: list,
        names: list,
        dist_func: callable,
        show_values: bool = True,
        remove_diagonal: bool = True
):
    """
    Show the distance matrix for a list of clusters
    :param clusters: The list of clusters to use
    :param names: The names for the rows and columns of the distance matrix
    :param dist_func: The distance function to use
    :param show_values: Whether to show the values in each cell
    :param remove_diagonal: Whether to remove the values on the diagonal (compare with itself)
    :return: The data for the distance matrix, the figure and the axis
    """
    # Initialize the distance matrix to 0
    num_clusters = len(clusters)
    distance_matrix = np.zeros(shape=(num_clusters, num_clusters))

    # Compute the distances above the diagonal

    def execution(row):
        for col in range(row + 1, num_clusters):
            lhs, rhs = clusters[row], clusters[col]
            distance_matrix[row][col] = dist_func(lhs, rhs)

    show_progress(execution=execution, iterable=range(0, num_clusters - 1))

    # Mirror on the diagonal to complete the rest of the matrix
    distance_matrix = distance_matrix + distance_matrix.T

    # Prepare the data for the image
    plot_data = pd.DataFrame(
        distance_matrix,
        columns=names,
        index=names
    )
    # Find the average value for each cell
    plot_data = plot_data.groupby(plot_data.columns, axis=1).mean()
    plot_data = plot_data.groupby(plot_data.index, axis=0).mean()
    # Remove the values on the diagonal
    if remove_diagonal:
        np.fill_diagonal(plot_data.values, np.nan)

    #  Show the image
    fig_size = 2 * len(plot_data)
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = sns.heatmap(
        plot_data,
        annot=show_values,
        cmap='OrRd',
        linewidth=.1,
        vmin=0, vmax=1
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    return plot_data, fig, ax
