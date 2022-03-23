import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from utils.dimensionality_reduction.dimensionality_reduction import tsne


def visualize_clusters(heatmaps: np.ndarray, clusters: np.ndarray, dim_red_technique=tsne, fig=None, ax=None,
                       marker='o'):
    # prepare the general figure
    if fig is None or ax is None:
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
        heatmaps_label = heatmaps[clusters == label]
        ax.plot(
            heatmaps_label[:, 0], heatmaps_label[:, 1],
            'x' if label == -1 else marker, markerfacecolor=tuple(color), markeredgecolor='k',
            label='no cluster' if label == -1 else label
        )
        plt.legend()

    return fig, ax
