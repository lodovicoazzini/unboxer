import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from utils.stats import compute_distance_matrix


def show_distance_matrix(
        values: list,
        index: list,
        dist_func: callable,
        show_values: bool = True,
        remove_diagonal: bool = True,
        values_range: tuple = (0, 1),
        show_color_bar: bool = True,
        show_progress_bar=False
):
    """
    Show the distance matrix for a list of clusters
    :param values: The list of clusters to use
    :param index: The names for the rows and columns of the distance matrix
    :param dist_func: The distance function to use
    :param show_values: Whether to show the values in each cell
    :param remove_diagonal: Whether to remove the values on the diagonal (compare with itself)
    :param values_range: The range for the values in the distance matrix
    :param show_color_bar: Whether to show the color-bar on the side
    :param show_progress_bar: Whether to show the progress bar
    :return: The data for the distance matrix, the figure and the axis
    """

    # Get the distance matrix
    dist_matrix = compute_distance_matrix(
        values=values,
        index=index,
        dist_func=dist_func,
        remove_diagonal=remove_diagonal,
        show_progress_bar=show_progress_bar
    )

    #  Show the image
    fig_size = len(dist_matrix)
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = sns.heatmap(
        dist_matrix,
        annot=show_values,
        cmap='OrRd',
        linewidth=.1,
        vmin=values_range[0] if values_range[0] is not None else np.nanmin(dist_matrix),
        vmax=values_range[1] if values_range[1] is not None else np.nanmax(dist_matrix),
        cbar=show_color_bar
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    return dist_matrix, fig, ax
