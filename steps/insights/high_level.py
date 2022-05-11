import numpy as np
import pandas as pd
import tensorflow as tf
from clusim.clustering import Clustering
from sklearn.manifold import TSNE

from config.config_dirs import FEATUREMAPS_DATA, PREDICTIONS
from config.config_featuremaps import FEATUREMAPS_CLUSTERS_MODE
from config.config_general import CLUS_SIM, CLUSTERS_SORT_METRIC, MAX_SAMPLES, MAX_LABELS
from utils.cluster.postprocessing import sorted_clusters
from utils.cluster.preprocessing import distance_matrix
from utils.cluster.visualize import visualize_clusters_projections, visualize_clusters_images
from utils.dataset import get_train_test_data, get_data_masks
from utils.general import save_figure, show_progress

BASE_DIR = f'out/featuremaps/{FEATUREMAPS_CLUSTERS_MODE.value}'


def featuremaps_distance_matrix():
    print('Reading the data ...')
    df = pd.read_pickle(FEATUREMAPS_DATA)
    df = df[df['mode'] == FEATUREMAPS_CLUSTERS_MODE.value]
    featuremaps_clusters = [
        Clustering().from_cluster_list(clusters_configuration)
        for clusters_configuration in df['clusters'].values
    ]
    df['complete_approach'] = df.apply(
        lambda row: f'{row["approach"]}({row["map_size"]})_{row["mode"]}',
        axis=1
    )

    print('Exporting the distance matrix for the featuremaps ...')
    dist_matrix, fig, ax = distance_matrix(
        featuremaps_clusters,
        lambda l, r: 1 - CLUS_SIM(l, r),
        show_map=True,
        names=df['complete_approach']
    )
    ax.set_title('Distance matrix for the feature combinations')
    save_figure(fig, f'{BASE_DIR}/distance_matrix')


def featuremaps_clusters_projections():
    print('Reading the data ...')
    df = pd.read_pickle(FEATUREMAPS_DATA)
    df = df[df['mode'] == FEATUREMAPS_CLUSTERS_MODE.value]
    clusters = [
        Clustering().from_cluster_list(clusters_configuration)
        for clusters_configuration in df['clusters'].values
    ]
    df['complete_approach'] = df.apply(
        lambda row: f'{row["approach"]}({row["map_size"]})_{row["mode"]}',
        axis=1
    )
    # Get the train and test data
    (train_data, train_labels), (test_data, test_labels) = get_train_test_data(rgb=True)
    train_data_gs, test_data_gs = (
        tf.image.rgb_to_grayscale(train_data).numpy(),
        tf.image.rgb_to_grayscale(test_data).numpy()
    )
    # Load the predictions
    predictions = np.loadtxt(PREDICTIONS)
    # Get the masks to filter the data
    mask_miss, mask_label = get_data_masks(test_labels, predictions, label=5)
    mask_miss_label = mask_miss[mask_label]

    print('Generating the projections ...')
    # Project the data in the 2d latent space
    projections = TSNE(perplexity=40).fit_transform(
        test_data_gs[mask_label].reshape(test_data_gs[mask_label].shape[0], -1)
    )

    print('Exporting the clusters projections ...')
    # Show the clusters projections for each feature combination
    zipped = list(zip(df['complete_approach'].values, clusters))
    show_progress(0, len(zipped))
    for idx, zipped_item in enumerate(zipped):
        feature_combination, clusters = zipped_item
        clusters_membership = np.array(clusters.to_membership_list())
        # Show the clusters projections
        fig, ax = visualize_clusters_projections(
            projections=projections,
            clusters=clusters_membership,
            mask=mask_miss_label
        )
        fig.suptitle(f'Clusters projections for the features {feature_combination}')
        save_figure(fig, f'{BASE_DIR}/clusters_projections_{feature_combination}')

        show_progress(idx, len(zipped))


def featuremaps_clusters_images():
    print('Reading the data ...')
    df = pd.read_pickle(FEATUREMAPS_DATA)
    df = df[df['mode'] == FEATUREMAPS_CLUSTERS_MODE.value]
    clusters = [
        Clustering().from_cluster_list(clusters_configuration)
        for clusters_configuration in df['clusters'].values
    ]
    df['complete_approach'] = df.apply(
        lambda row: f'{row["approach"]}({row["map_size"]})_{row["mode"]}',
        axis=1
    )
    # Get the train and test data
    (train_data, train_labels), (test_data, test_labels) = get_train_test_data(rgb=True)
    train_data_gs, test_data_gs = (
        tf.image.rgb_to_grayscale(train_data).numpy(),
        tf.image.rgb_to_grayscale(test_data).numpy()
    )
    # Load the predictions
    predictions = np.loadtxt(PREDICTIONS)
    # Get the masks to filter the data
    mask_miss, mask_label = get_data_masks(test_labels, predictions, label=5)
    mask_miss_label = mask_miss[mask_label]

    print('Exporting the clusters sample images ...')
    # Show the clusters projections for each feature combination
    zipped = list(zip(df['complete_approach'].values, clusters))
    show_progress(0, len(zipped))
    for idx, zipped_item in enumerate(zipped):
        feature_combination, clusters = zipped_item
        clusters_membership = np.array(clusters.to_membership_list())
        # Get the mask for the clusters containing misclassified elements of the selected label
        mask_contains_miss_label = np.isin(clusters_membership, np.unique(clusters_membership[mask_miss_label]))

        # sample some clusters labels containing misclassified items
        unique_labels = np.unique(clusters_membership[mask_contains_miss_label])
        sample_labels = np.random.choice(unique_labels, min(4, len(unique_labels)), replace=False)
        sample_mask = np.isin(clusters_membership, sample_labels)
        # Sort the clusters
        if CLUSTERS_SORT_METRIC is not None:
            clusters_list = clusters.to_cluster_list()
            clusters_list = sorted_clusters(clusters_list, metric=CLUSTERS_SORT_METRIC)
            clusters_membership = np.array(Clustering().from_cluster_list(clusters_list).to_membership_list())
        # show some correctly classified images for clusters containing also misclassified images
        fig, _ = visualize_clusters_images(
            clusters=clusters_membership[mask_contains_miss_label & ~mask_miss_label & sample_mask],
            images=test_data_gs[mask_label][mask_contains_miss_label & ~mask_miss_label & sample_mask],
            predictions=predictions[mask_label][mask_contains_miss_label & ~mask_miss_label & sample_mask],
            max_samples=MAX_SAMPLES,
            max_labels=MAX_LABELS,
            cmap='gray_r'
        )
        fig.suptitle(f'Correctly classified items for {feature_combination}')
        save_figure(fig, f'{BASE_DIR}/correct_samples_{feature_combination}')
        # show some incorrectly classified images for clusters containing also misclassified images
        fig, _ = visualize_clusters_images(
            clusters=clusters_membership[mask_contains_miss_label & mask_miss_label & sample_mask],
            images=test_data_gs[mask_label][mask_contains_miss_label & mask_miss_label & sample_mask],
            predictions=predictions[mask_label][mask_contains_miss_label & mask_miss_label & sample_mask],
            max_samples=MAX_SAMPLES,
            max_labels=MAX_LABELS,
            cmap='gray_r'
        )
        fig.suptitle(f'Misclassified classified items for {feature_combination}')
        save_figure(fig, f'{BASE_DIR}/misclassified_samples_{feature_combination}')

        show_progress(idx, len(zipped))
