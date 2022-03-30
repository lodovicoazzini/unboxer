import math

import matplotlib.pyplot as plt
import numpy as np


def visualize_images_grid(
        images: np.ndarray,
        real_labels: np.ndarray,
        predicted_labels: np.ndarray,
        grid_size: int = 4,
        fig_size: tuple[int, int] = (8, 8),
        cmap='gray_r',
        show_legend: bool = True
):
    # compute the number of rows and cols
    n_cols = grid_size
    n_rows = math.ceil(len(images) / grid_size)
    # create the overall figure
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(fig_size[0] * n_cols, fig_size[1] * n_rows))

    # keep track of the last visualized image for the legend
    last_img = None
    # associate each slot in the matrix to an image
    for ax_i, image, real, predicted in zip(ax.flatten(), images, real_labels, predicted_labels):
        last_img = ax.imshow(np.ma.masked_equal(image, 0).filled(np.nan), cmap=cmap)
        ax.set_title(f'Real: {real} Predicted: {predicted}')

    # remove the axis
    for ax_i in ax.flatten():
        ax_i.axis('off')

    # add the legend
    if last_img is not None and show_legend:
        fig.subplots_adjust(right=0.89)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(last_img, cax=cbar_ax)
        cbar.set_ticks([])

    return fig, ax


def visualize_images_by_label(
        images: np.ndarray,
        real_labels: np.ndarray,
        predicted_labels: np.ndarray,
        max_images: int = None, max_labels: int = None,
        fig_size: tuple[int, int] = (8, 8),
        cmap: str = 'gray_r',
        show_legend: bool = True
):
    # rows = unique labels or provided value if smaller
    # cols = max samples for label or provided value if smaller
    real_labels_unique, counts_per_label = np.unique(real_labels, return_counts=True)
    n_rows = real_labels_unique.shape[0] if max_labels is None else min(real_labels_unique.shape[0], max_labels)
    n_cols = np.amax(counts_per_label) if max_images is None else min(np.amax(counts_per_label), max_images)
    # generate the overall image
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(fig_size[0] * max_images, fig_size[1] * max_labels))

    # if the provided number of labels is less than the actual labels -> sample
    labels_sample = real_labels_unique
    if max_labels is not None and real_labels_unique.shape[0] > max_labels:
        labels_sample = np.random.choice(
            real_labels_unique,
            max_labels,
            replace=False
        )

    # keep track of the last visualized image for the legend
    last_img = None
    # iterate through the labels
    for row, label in enumerate(sorted(labels_sample)):
        # filter for the selected label
        images_filtered = images[real_labels == label]
        # if the number of filtered images is greater than the provided value -> sample
        images_sample = images_filtered
        real_labels_sample = real_labels
        predicted_labels_sample = predicted_labels
        if max_images is not None and images_filtered.shape[0] > max_images:
            sample_idxs = np.random.choice(images_filtered.shape[0], max_images, replace=False)
            images_sample = images_filtered[sample_idxs]
            real_labels_sample = real_labels[sample_idxs]
            predicted_labels_sample = predicted_labels[sample_idxs]

        for col, zipped in enumerate(zip(images_sample, real_labels_sample, predicted_labels_sample)):
            image, real, predicted = zipped
            last_img = ax.imshow(np.ma.masked_equal(image, 0).filled(np.nan), cmap=cmap)
            ax.set_title(f'Real: {real} Predicted: {predicted}')

    # remove the axis
    for ax_i in ax.flatten():
        ax_i.axis('off')

    # add the legend
    if last_img is not None and show_legend:
        fig.subplots_adjust(right=0.89)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(last_img, cax=cbar_ax)
        cbar.set_ticks([])

    return fig, ax


def visualize_distance_matrix(
        distance_matrix: np.ndarray,
        images: np.ndarray,
        cell_size: tuple[int, int] = (8, 8),
        images_cmap='gray_r',
        cells_cmap='inferno',
        text_size=60
):
    # create the overall figure
    n_rows = distance_matrix.shape[0] + 1
    n_cols = distance_matrix.shape[1] + 1
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(cell_size[0] * n_cols, cell_size[1] * n_rows))
    # plot the borders of the matrix
    for idx, pos in enumerate(range(1, n_rows)):
        ax[pos][0].imshow(images[idx], cmap=images_cmap)
        ax[0][pos].imshow(images[idx], cmap=images_cmap)

    # create the colormap
    # keep only the distances between different heatmaps
    diff_distances = distance_matrix[np.triu_indices(len(distance_matrix), 1)]
    # normalize the values in [0, 1]
    dist_matrix_norm = (
            (distance_matrix - np.amin(diff_distances)) /
            (np.amax(distance_matrix) - np.amin(diff_distances)))
    cmap = plt.cm.get_cmap(cells_cmap)

    # visualize all the distances
    for r_idx, row in enumerate(range(1, n_rows)):
        for c_idx, col in enumerate(range(1, n_cols)):
            # skip the diagonal (same heatmap)
            if row != col:
                # color the cell
                ax[row][col].set(facecolor=cmap(dist_matrix_norm[r_idx][c_idx]))
                # add the distance value
                ax[row][col].text(
                    .5, .5,
                    round(distance_matrix[r_idx][c_idx], 3),
                    horizontalalignment='center',
                    verticalalignment='center',
                    size=text_size
                )

    return fig, ax
