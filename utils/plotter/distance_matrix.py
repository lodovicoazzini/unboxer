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
        remove_diagonal: bool = True,
        values_range: tuple = (0, 1),
        show_color_bar: bool = True
):
    """
    Show the distance matrix for a list of clusters
    :param clusters: The list of clusters to use
    :param names: The names for the rows and columns of the distance matrix
    :param dist_func: The distance function to use
    :param show_values: Whether to show the values in each cell
    :param remove_diagonal: Whether to remove the values on the diagonal (compare with itself)
    :param values_range: The range for the values in the distance matrix
    :param show_color_bar: Whether to show the color-bar on the side
    :return: The data for the distance matrix, the figure and the axis
    """
    # Initialize the distance matrix to 0
    num_clusters = len(clusters)
    distance_matrix = np.zeros(shape=(num_clusters, num_clusters))

    # Compute the distances above the diagonal
    def execution(row):
        for col in range(row, num_clusters):
            lhs, rhs = clusters[row], clusters[col]
            distance_matrix[row][col] = dist_func(lhs, rhs)

    show_progress(execution=execution, iterable=range(0, num_clusters))

    # Mirror on the diagonal to complete the rest of the matrix
    distance_matrix = distance_matrix + distance_matrix.T
    distance_matrix[np.diag_indices_from(distance_matrix)] /= 2

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
    fig_size = len(plot_data)
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = sns.heatmap(
        plot_data,
        annot=show_values,
        cmap='OrRd',
        linewidth=.1,
        vmin=values_range[0] if values_range[0] is not None else np.nanmin(plot_data.values),
        vmax=values_range[1] if values_range[1] is not None else np.nanmax(plot_data.values),
        cbar=show_color_bar
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    return plot_data, fig, ax
