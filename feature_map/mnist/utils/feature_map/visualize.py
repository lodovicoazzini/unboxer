import itertools
import time

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from config.config_featuremaps import NUM_CELLS
from config.config_general import EXPECTED_LABEL
from feature_map.mnist.utils.feature_map.compute import compute_map
from utils.general import save_figure


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
    for feature1, feature2 in itertools.combinations(features, 2):
        start_time = time.time()
        features_comb = [feature1, feature2]
        _, coverage_data, misbehaviour_data, clusters = compute_map(features_comb, samples)

        # figure
        fig, ax = plt.subplots(figsize=(8, 8))

        cmap = sns.cubehelix_palette(dark=0.5, light=0.9, as_cmap=True)
        # Set the color for the under the limit to be white (so they are not visualized)
        cmap.set_under('1.0')

        # For some weird reason the data in the heatmap are shown with the first dimension on the y and the
        # second on the x. So we transpose
        coverage_data = np.transpose(coverage_data)

        sns.heatmap(coverage_data, vmin=1, vmax=20, square=True, cmap=cmap)

        # Plot misbehaviors - Iterate over all the elements of the array to get their coordinates:
        it = np.nditer(misbehaviour_data, flags=['multi_index'])
        for v in it:
            # Plot only misbehaviors
            if v > 0:
                alpha = 0.1 * v if v <= 10 else 1.0
                (x, y) = it.multi_index
                # Plot as scattered plot. the +0.5 ensures that the marker in centered in the cell
                plt.scatter(x + 0.5, y + 0.5, color="black", alpha=alpha, s=50)

        xtickslabel = [round(the_bin, 1) for the_bin in feature1.get_bins_labels()]
        ytickslabel = [round(the_bin, 1) for the_bin in feature2.get_bins_labels()]
        ax.set_xticklabels(xtickslabel)
        plt.xticks(rotation=45)
        ax.set_yticklabels(ytickslabel)
        plt.yticks(rotation=0)
        fig.suptitle(f'Feature map: digit {EXPECTED_LABEL}', fontsize=16)
        # Plot small values of y below.
        # We need this to have the y axis start from zero at the bottom
        ax.invert_yaxis()
        # axis labels
        plt.xlabel(feature1.feature_name)
        plt.ylabel(feature2.feature_name)

        features_comb_str = '+'.join([feature.feature_name for feature in [feature1, feature2]])
        data.append({
            'approach': features_comb_str,
            'map_size': NUM_CELLS,
            'map_time': time.time() - start_time,
            'clusters': clusters
        })

        map_size_str = f'{NUM_CELLS}x{NUM_CELLS}'
        save_figure(fig, f'out/feature_maps/featuremap_{EXPECTED_LABEL}_{map_size_str}_{features_comb_str}')

    return data
