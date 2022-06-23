from typing import Callable

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from utils.stats import compute_comparison_matrix


def show_comparison_matrix(
        values: list,
        index: list,
        metric: Callable,
        show_values: bool = True,
        remove_diagonal: bool = True,
        values_range: tuple = (0, 1),
        show_color_bar: bool = True,
        show_progress_bar=False,
        multi_process: bool = False
):
    """
    Show the distance matrix for a list of clusters
    :param values: The list of clusters to use
    :param index: The names for the rows and columns of the distance matrix
    :param metric: The distance function to use
    :param show_values: Whether to show the values in each cell
    :param remove_diagonal: Whether to remove the values on the diagonal (compare with itself)
    :param values_range: The range for the values in the distance matrix
    :param show_color_bar: Whether to show the color-bar on the side
    :param show_progress_bar: Whether to show the progress bar
    :param multi_process: Whether to use multiple processes to compute the matrix
    :return: The data for the distance matrix, the figure and the axis
    """

    # Get the distance matrix
    matrix = compute_comparison_matrix(
        values=values,
        metric=metric,
        show_progress_bar=show_progress_bar,
        multi_process=multi_process
    )
    if remove_diagonal:
        np.fill_diagonal(matrix, np.nan)

    #  Show the image
    fig_size = len(matrix)
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = sns.heatmap(
        matrix,
        annot=show_values,
        xticklabels=index,
        yticklabels=index,
        cmap='OrRd',
        linewidth=.1,
        vmin=values_range[0] if values_range[0] is not None else np.nanmin(matrix),
        vmax=values_range[1] if values_range[1] is not None else np.nanmax(matrix),
        cbar=show_color_bar
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    return matrix, fig, ax
