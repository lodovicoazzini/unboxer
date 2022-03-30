import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def visualize_clusters_projections(
        projections: np.ndarray,
        clusters: np.ndarray,
        fig=None, ax=None,
        marker='o', size=30, edge_color='#000000',
        cmap='tab10', label_prefix: str = None
):
    # prepare the general figure if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))
    # remove the axis, ticks and labels
    sns.despine(left=True, bottom=True)
    ax.tick_params(left=False, bottom=False)
    ax.set(xticklabels=[], yticklabels=[])

    # associate one cluster label for each color
    cluster_labels = sorted(set(clusters))
    colors = [plt.cm.get_cmap(cmap)(val) for val in np.linspace(0, 1, len(cluster_labels))]
    colors_dict = {label: colors[idx] for idx, label in enumerate(cluster_labels)}
    # associate black with noise
    if -1 in cluster_labels:
        colors_dict[-1] = (0, 0, 0, 1)

    for label, color in colors_dict.items():
        # plot the core samples
        projections_filtered = projections[clusters == label]
        if label == -1:
            label_str = 'no cluster'
        elif label_prefix is not None:
            label_str = '_'.join([label_prefix, str(label)])
        else:
            label_str = str(label)
        ax.scatter(
            projections_filtered[:, 0], projections_filtered[:, 1],
            marker='x' if label == -1 else marker, c=[color], edgecolors=edge_color,
            label=label_str,
            s=size
        )
        # add the legend
        ax.legend()

    return fig, ax


def visualize_clusters_images(
        clusters: np.ndarray,
        images: np.ndarray,
        predictions: np.ndarray,
        max_labels: int = None, max_samples: int = None,
        fig_size: int = 8, cmap: str = 'gray',
        show_legend: bool = True
):
    # get the individual labels
    labels = np.unique(clusters)
    # sample the clusters by labels if a maximum number of labels is provided
    clusters_sample = clusters
    images_sample = images
    predictions_sample = predictions
    if max_labels is not None and labels.shape[0] > max_labels:
        sample_mask = np.isin(clusters, np.random.choice(labels, max_labels, replace=False))
        clusters_sample = clusters[sample_mask]
        images_sample = images[sample_mask]
        predictions_sample = predictions[sample_mask]

    # the number of rows is the number of selected labels
    labels_sample, items_count = np.unique(clusters_sample, return_counts=True)
    n_rows = labels_sample.shape[0]
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
            size=60
        )
        # plot the heatmaps corresponding to the label
        images_filtered = images_sample[clusters_sample == label]
        label_predictions = predictions_sample[clusters_sample == label]
        # if the number of heatmaps is greater than the provided value -> draw a random sample
        images_filtered_sample = images_filtered
        label_predictions_sample = label_predictions
        if images_filtered.shape[0] > max_samples:
            sample_idxs = np.random.choice(images_filtered.shape[0], max_samples, replace=False)
            images_filtered_sample = images_filtered[sample_idxs]
            label_predictions_sample = label_predictions[sample_idxs]
        for col, zipped in enumerate(zip(images_filtered_sample, label_predictions_sample)):
            image, prediction = zipped
            last_image = ax[row][col + 1].imshow(np.ma.masked_equal(image, 0).filled(np.nan), cmap=cmap)
            ax[row][col + 1].set_title(f'Prediction: {int(prediction)}')
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

    return fig, ax
