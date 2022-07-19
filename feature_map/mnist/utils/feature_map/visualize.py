import time
from functools import reduce
from itertools import combinations

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from config.config_data import EXPECTED_LABEL
from config.config_featuremaps import NUM_CELLS, MAP_DIMENSIONS
from feature_map.mnist.utils.feature_map.compute import compute_map
from utils.general import save_figure, scale_values_in_range


def visualize_map(features, samples):
    """
        Visualize the samples and the features on a map. The map cells contains the number of samples for each
        cell, so empty cells (0) are white, cells with few elements have a light color, while cells with more
        elements have darker color. This gives an intuition on the distribution of the misbheaviors and the
        collisions
    Returns:
    """
    print('Generating the featuremaps ...')

    # Create one visualization for each pair of self.axes selected in order
    data = []
    map_dimensions = min(MAP_DIMENSIONS, 3)
    # Compute all the 2d and 3d feature combinations
    features_combinations = reduce(
        lambda acc, comb: acc + comb,
        [
            list(combinations(features, n_features))
            for n_features in list(range(2, min(map_dimensions + 1, 4)))]
    )
    for features_combination in features_combinations:
        start_time = time.time()
        features_comb_str = '+'.join([feature.feature_name for feature in features_combination])
        map_size_str = f'{NUM_CELLS}x{NUM_CELLS}'
        # Place the values over the map
        _, coverage_data, misbehavior_data, clusters = compute_map(features_combination, samples)
        # Handle the case of 3d maps
        if len(features_combination) == 3:
            # Visualize the map
            fig, ax = visualize_3d_map(coverage_data, misbehavior_data)
        # Handle the case of 2d maps
        else:
            # Visualize the map
            fig, ax = visualize_2d_map(coverage_data, misbehavior_data)

        # Set the style
        fig.suptitle(f'Feature map: digit {EXPECTED_LABEL}', fontsize=16)
        ax.set_xlabel(features_combination[0].feature_name)
        ax.set_ylabel(features_combination[1].feature_name)
        if len(features_combination) == 3:
            ax.set_zlabel(features_combination[2].feature_name)
        # Export the figure
        save_figure(fig, f'out/featuremaps/featuremap_{EXPECTED_LABEL}_{map_size_str}_{features_comb_str}')

        # Record the data
        data.append({
            'approach': features_comb_str,
            'map_size': NUM_CELLS,
            'map_time': time.time() - start_time,
            'clusters': clusters
        })
    plt.show()
    return data


def visualize_3d_map(coverage_data, misbehavior_data):
    # Create the figure
    fig = plt.figure(figsize=(8, 8))
    # Set the 3d plot
    ax = fig.add_subplot(projection='3d')
    # Get the coverage and misbehavior data for the plot
    x_coverage, y_coverage, z_coverage, values_coverage = unpack_plot_data(coverage_data)
    x_misbehavior, y_misbehavior, z_misbehavior, values_misbehavior = unpack_plot_data(misbehavior_data)
    sizes_coverage, sizes_misbehavior = scale_values_in_range([values_coverage, values_misbehavior], 100, 1000)
    # Plot the data
    ax.scatter(x_coverage, y_coverage, z_coverage, s=sizes_coverage, alpha=.4, label='all')
    ax.scatter(x_misbehavior, y_misbehavior, z_misbehavior, s=sizes_misbehavior, c='red', alpha=.8,
               label='misclassified')
    ax.legend(markerscale=.3, frameon=False, bbox_to_anchor=(.3, 1.1))
    return fig, ax


def visualize_2d_map(coverage_data, misbehavior_data):
    # Compute the percentage of misbehavior
    misbehavior_data = misbehavior_data / coverage_data
    np.nan_to_num(misbehavior_data, copy=False, nan=0)
    # The heatmap inverts x and y -> transpose
    coverage_data = np.transpose(coverage_data)
    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 8))
    # Set the colormap
    colormap = sns.cubehelix_palette(dark=0.5, light=0.9, as_cmap=True)
    # Set the color for out-of-rage values to be white (not visible)
    colormap.set_under('1.0')
    # Plot the coverage data
    sns.heatmap(coverage_data, vmin=1, vmax=20, square=True, cmap=colormap, cbar_kws={'label': 'cluster size'})
    # Plot the misbehavior data
    x, y, values = unpack_plot_data(misbehavior_data)
    # Ensure that the markers are centered in the cells
    plt.scatter(x + .5, y + .5, color="black", alpha=values, s=100, label='misclassified (%)')
    # Style
    ax.legend(frameon=False, bbox_to_anchor=(.3, 1.1))
    return fig, ax


def unpack_plot_data(data: np.ndarray):
    # Get the indexes where the values are not zero
    indexes = [np.array(idx) for idx in data.nonzero()]
    # Get the values corresponding to each triple of indexes
    values = np.array([data[idx] for idx in zip(*indexes)])
    return *indexes, values
