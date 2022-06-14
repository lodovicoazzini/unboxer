import numpy as np
import pandas as pd
from clusim.clustering import Clustering
from sklearn.manifold import TSNE

from config.config_dirs import FEATUREMAPS_DATA
from config.config_featuremaps import FEATUREMAPS_CLUSTERING_MODE
from config.config_general import CLUSTERS_SIMILARITY_METRIC, CLUSTERS_SORT_METRIC
from config.config_outputs import MAX_LABELS
from utils import global_values
from utils.clusters.postprocessing import get_sorted_clusters
from utils.general import save_figure, show_progress
from utils.plotter.distance_matrix import show_distance_matrix
from utils.plotter.visualize import show_clusters_projections, visualize_clusters_images

BASE_DIR = f'out/featuremaps/{FEATUREMAPS_CLUSTERING_MODE.value}'


def featuremaps_distance_matrix():
    print('Exporting the distance matrix for the featuremaps ...')
    # Read the data for the featuremaps
    df = pd.read_pickle(FEATUREMAPS_DATA)
    # Filter for the selected clustering mode
    df = df[df['mode'] == FEATUREMAPS_CLUSTERING_MODE.value]
    # Get the list of clusters configurations
    featuremaps_clusters = [
        Clustering().from_cluster_list(clusters_configuration)
        for clusters_configuration in df['clusters'].values
    ]
    # Merge the information about the approach
    df['complete_approach'] = df.apply(
        lambda row: f'{row["approach"]}({row["map_size"]})_{row["mode"]}',
        axis=1
    )
    dist_matrix, fig, ax = show_distance_matrix(
        featuremaps_clusters,
        dist_func=lambda l, r: 1 - CLUSTERS_SIMILARITY_METRIC(l, r),
        index=df['complete_approach'],
        show_progress_bar=True,
        remove_diagonal=False
    )
    ax.set_title('Distance matrix for the feature combinations')
    save_figure(fig, f'{BASE_DIR}/distance_matrix')


def featuremaps_clusters_projections():
    print('Reading the data ...')
    df = pd.read_pickle(FEATUREMAPS_DATA)
    df = df[df['mode'] == FEATUREMAPS_CLUSTERING_MODE.value]
    clusters = [
        Clustering().from_cluster_list(clusters_configuration)
        for clusters_configuration in df['clusters'].values
    ]
    df['complete_approach'] = df.apply(
        lambda row: f'{row["approach"]}({row["map_size"]})_{row["mode"]}',
        axis=1
    )

    print('Generating the projections ...')
    # Project the data in the 2d latent space
    projections = TSNE(perplexity=40).fit_transform(
        global_values.test_data_gs[global_values.mask_label].reshape(
            global_values.test_data_gs[global_values.mask_label].shape[0],
            -1
        )
    )

    print('Exporting the clusters projections ...')
    # Show the clusters projections for each feature combination
    zipped = list(zip(df['complete_approach'].values, clusters))

    def execution(feature_combination, cluster_configuration):
        clusters_membership = np.array(cluster_configuration.to_membership_list())
        # Show the clusters projections
        fig, ax = show_clusters_projections(
            projections=projections,
            cluster_membership=clusters_membership,
            mask=global_values.mask_miss_label
        )
        fig.suptitle(f'Clusters projections for the features {feature_combination}')
        save_figure(fig, f'{BASE_DIR}/clusters_projections_{feature_combination}')

    show_progress(execution=execution, iterable=zipped)


def featuremaps_clusters_images():
    print('Reading the data ...')
    df = pd.read_pickle(FEATUREMAPS_DATA)
    df = df[df['mode'] == FEATUREMAPS_CLUSTERING_MODE.value]
    df['complete_approach'] = df.apply(
        lambda row: f'{row["approach"]}({row["map_size"]})_{row["mode"]}',
        axis=1
    )

    print('Exporting the clusters sample images ...')
    # Show the clusters projections for each feature combination
    tuples = df[['complete_approach', 'clusters']].values

    def execution(approach, clusters):
        # Get the mask for the clusters containing misclassified elements of the selected label
        clusters_membership = np.array(Clustering().from_cluster_list(clusters).to_membership_list())
        labels_contains_miss = np.unique(clusters_membership[global_values.mask_miss_label])
        mask_contains_miss_label = np.isin(clusters_membership, labels_contains_miss)
        # Sample some clusters containing misclassified entries
        sample_labels = np.random.choice(
            labels_contains_miss,
            min(MAX_LABELS, len(labels_contains_miss)),
            replace=False
        )
        mask_contains_sample_label = np.isin(clusters_membership, sample_labels)
        # Sort the clusters
        if CLUSTERS_SORT_METRIC is not None:
            clusters_list = get_sorted_clusters(clusters, metric=CLUSTERS_SORT_METRIC)
            clusters_membership = np.array(Clustering().from_cluster_list(clusters_list).to_membership_list())
        # show some correctly classified images for clusters containing also misclassified images
        correct_sample_mask = mask_contains_miss_label & ~global_values.mask_miss_label & mask_contains_sample_label
        fig, _ = visualize_clusters_images(
            cluster_membership=clusters_membership[correct_sample_mask],
            images=global_values.test_data_gs[global_values.mask_label][correct_sample_mask],
            predictions=global_values.predictions[global_values.mask_label][correct_sample_mask]
        )
        fig.suptitle(f'Correctly classified items for {approach}')
        save_figure(fig, f'{BASE_DIR}/correct_samples_{approach}')
        # show some incorrectly classified images for clusters containing also misclassified images
        misses_sample_mask = mask_contains_miss_label & global_values.mask_miss_label & mask_contains_sample_label
        fig, _ = visualize_clusters_images(
            cluster_membership=clusters_membership[misses_sample_mask],
            images=global_values.test_data_gs[global_values.mask_label][misses_sample_mask],
            predictions=global_values.predictions[global_values.mask_label][misses_sample_mask]
        )
        fig.suptitle(f'Misclassified classified items for {approach}')
        save_figure(fig, f'{BASE_DIR}/misclassified_samples_{approach}')

    show_progress(execution=execution, iterable=tuples)
