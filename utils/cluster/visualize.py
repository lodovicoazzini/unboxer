import math

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def visualize_clusters_projections(
        projections: np.ndarray,
        clusters: np.ndarray,
        mask
):
    # Prepare the general figure if not provided
    fig = plt.figure(figsize=(16, 9))
    # Remove the axis, ticks and labels
    sns.despine(left=True, bottom=True)

    # Plot the data
    ax = sns.scatterplot(
        x=[projection[0] for projection in projections],
        y=[projection[1] for projection in projections],
        hue=clusters,
        style=['misclassified' if v else 'correct' for v in mask],
        palette=sns.color_palette('viridis', n_colors=len(set(clusters)))
    )
    # Style
    ax.tick_params(left=False, bottom=False)
    ax.set(xticklabels=[], yticklabels=[])
    ax.legend(ncol=5)

    return fig, ax


def visualize_clusters_images(
        clusters: np.ndarray,
        images: np.ndarray,
        predictions: np.ndarray,
        overlay: np.ndarray = None,
        max_labels: int = None, max_samples: int = None,
        fig_size: int = 8, cmap: str = 'gray', overlay_cmap='Reds', overlay_alpha=.7,
        show_legend: bool = True,
        label_size: int = 60, titles_size: int = 60
):
    # get the individual labels
    labels = np.unique(clusters)
    # sample the clusters by labels if a maximum number of labels is provided
    clusters_sample = clusters
    images_sample = images
    predictions_sample = predictions
    overlay_sample = overlay if overlay is not None else np.empty_like(images)
    if max_labels is not None and labels.shape[0] > max_labels:
        sample_mask = np.isin(clusters, np.random.choice(labels, max_labels, replace=False))
        clusters_sample = clusters[sample_mask]
        images_sample = images[sample_mask]
        predictions_sample = predictions[sample_mask]
        overlay_sample = overlay[sample_mask]

    # If no cluster is in the sample return an empty image
    if len(clusters_sample) == 0:
        fig, ax = plt.subplots(1, 1)
        ax.text(
            .5, .5,
            'EMPTY SAMPLE',
            horizontalalignment='center',
            verticalalignment='center',
            size=label_size
        )
        return fig, ax
    # the number of rows is the number of selected labels
    labels_sample, items_count = np.unique(clusters_sample, return_counts=True)
    # Add one row to prevent errors and remove it at the end
    n_rows = labels_sample.shape[0] + 1
    # the number of columns is the number of selected individuals + 1 for the label
    n_cols = 1 + np.amax(items_count)
    if max_samples is not None:
        n_cols = 1 + min(max_samples, np.amax(items_count))
    # generate the overall image
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(fig_size * n_cols, fig_size * n_rows))

    # keep track of the last image for the legend
    last_image = None
    # for each label pot the corresponding images
    for row, label in enumerate(labels_sample):
        # show the labels in the first column
        ax[row][0].text(
            .5, .5,
            label,
            horizontalalignment='center',
            verticalalignment='center',
            size=label_size
        )
        # plot the heatmaps corresponding to the label
        images_filtered = images_sample[clusters_sample == label]
        label_predictions = predictions_sample[clusters_sample == label]
        overlay_filtered = overlay_sample[clusters_sample == label]
        # if the number of heatmaps is greater than the provided value -> draw a random sample
        images_filtered_sample = images_filtered
        label_predictions_sample = label_predictions
        overlay_filtered_sample = overlay_filtered
        if max_samples is not None and images_filtered.shape[0] > max_samples:
            sample_idxs = np.random.choice(images_filtered.shape[0], max_samples, replace=False)
            images_filtered_sample = images_filtered[sample_idxs]
            label_predictions_sample = label_predictions[sample_idxs]
            overlay_filtered_sample = overlay_filtered[sample_idxs]
        for col, zipped in enumerate(zip(images_filtered_sample, label_predictions_sample, overlay_filtered_sample)):
            image, prediction, overlay_image = zipped
            ax[row][col + 1].imshow(np.ma.masked_equal(image, 0).filled(np.nan), cmap=cmap)
            if overlay is not None:
                last_image = ax[row][col + 1].imshow(
                    np.ma.masked_equal(overlay_image, 0).filled(np.nan),
                    cmap=overlay_cmap,
                    alpha=overlay_alpha
                )
            ax[row][col + 1].set_title(f'Prediction: {int(prediction)}', size=titles_size)
            ax[row][col + 1].axis('off')

    # remove the ticks and labels
    for axx in ax.flatten():
        axx.axis('off')

    # add the legend
    if last_image is not None and show_legend:
        fig.subplots_adjust(right=0.89)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(last_image, cax=cbar_ax)
        cbar.set_ticks([])

    # Remove the last row
    for ax_i in ax[-1]:
        fig.delaxes(ax_i)

    return fig, ax


def visualize_cluster_images(
        cluster,
        images,
        overlays=None,
        predictions=None,
        cmap='gray_r',
        overlay_cmap='Reds',
        overlay_alpha=.7,
        imgs_per_row=6,
        max_images=36
):
    # Sample the images if too many
    if len(cluster) > max_images and predictions is not None:
        # Sample some misclassified indexes
        miss_idxs = np.argwhere(predictions != 5).flatten()
        sample_idxs = np.intersect1d(cluster, miss_idxs)
        if len(sample_idxs) < max_images:
            remaining_idxs = np.setdiff1d(cluster, sample_idxs)
            sample_idxs = np.concatenate((sample_idxs, np.random.choice(remaining_idxs, max_images - len(sample_idxs))))
    else:
        sample_idxs = cluster

    # Get the number of rows and columns
    if len(sample_idxs) < imgs_per_row:
        cols = len(sample_idxs)
        rows = 1
    else:
        cols = imgs_per_row
        rows = math.ceil(len(sample_idxs) / imgs_per_row)

    # Create the figure
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 8, rows * 8))

    # Sort the images by label
    if predictions is not None:
        cluster_labels = [predictions[idx] for idx in sample_idxs]
        sorted_cluster = sample_idxs[np.argsort(cluster_labels)]
    else:
        sorted_cluster = sample_idxs

    # Assign each axis to one index
    for axx, idx in zip(ax.flatten(), sorted_cluster):
        axx.imshow(np.ma.masked_equal(images[idx], 0).filled(np.nan), cmap=cmap)
        if overlays is not None:
            axx.imshow(np.ma.masked_equal(overlays[idx], 0).filled(np.nan), cmap=overlay_cmap, alpha=overlay_alpha)
        if predictions is not None:
            axx.set_title(f'Prediction: {int(predictions[idx])}', size=60)

    # remove the ticks and labels
    for axx in ax.flatten():
        axx.axis('off')

    return fig, ax
