import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from clusim.clustering import Clustering
from matplotlib import pyplot as plt

from config.config_dirs import HEATMAPS_DATA, HEATMAPS_DATA_RAW, PREDICTIONS
from config.config_general import EXPECTED_LABEL, CLUSTERS_SORT_METRIC, MAX_SAMPLES, MAX_LABELS, CLUS_SIM
from utils.cluster.postprocessing import sorted_clusters
from utils.cluster.preprocessing import distance_matrix
from utils.cluster.sample import sample_most_popular
from utils.cluster.visualize import visualize_clusters_projections, visualize_clusters_images
from utils.dataset import get_train_test_data, get_data_masks
from utils.general import save_figure, show_progress


def heatmaps_distance_matrix():
    # Read the data
    df = pd.read_pickle(HEATMAPS_DATA)

    print('Computing the distance matrix for the heatmaps ...')
    # Remove the configurations with only one cluster
    plot_data = df[df['num_clusters'] > 1]
    distances_df, fig, ax = distance_matrix(
        heatmaps=[Clustering().from_cluster_list(clusters) for clusters in plot_data['clusters']],
        dist_func=lambda lhs, rhs: 1 - CLUS_SIM(lhs, rhs),
        names=plot_data['explainer'],
        show_map=True
    )
    fig.suptitle('Distance matrix for the low-level approaches')
    save_figure(fig, f'out/low_level/distance_matrix')


def heatmaps_clusters_projections():
    # Read the data
    df = pd.read_pickle(HEATMAPS_DATA)
    predictions = np.loadtxt(PREDICTIONS)
    _, (test_data, test_labels) = get_train_test_data(rgb=True, verbose=False)
    mask_miss, mask_label = get_data_masks(real=test_labels, predictions=predictions, label=EXPECTED_LABEL,
                                           verbose=False)
    # Get the mask for the misclassified items of the selected label
    mask_miss_label = mask_miss[mask_label]

    # Get the most popular configurations
    df = sample_most_popular(df, group_by='explainer').set_index('explainer')
    # Iterate through the explainers
    print('Exporting the clusters projections ...')
    explainers = df.index.unique()
    show_progress(0, len(explainers))
    for idx, explainer in enumerate(explainers):
        # Get the best configuration for the explainer
        pick_config = df.loc[explainer]
        clusters, projections, contributions = pick_config[[
            'clusters',
            'projections',
            'contributions'
        ]]
        # Convert the clusters to membership list
        clusters_membership = np.array(Clustering().from_cluster_list(clusters).to_membership_list())

        # Visualize the projections of the contributions clusters
        fig, ax = visualize_clusters_projections(
            projections=projections,
            clusters=clusters_membership,
            mask=mask_miss_label
        )
        ax.set_title(f'{explainer} clusters projections')
        save_figure(fig, f'out/low_level/{explainer}/clusters_projections')

        show_progress(idx, len(explainers))
    print()


def heatmaps_clusters_images():
    # Read the data
    df = pd.read_pickle(HEATMAPS_DATA)
    predictions = np.loadtxt(PREDICTIONS)
    (train_data, train_labels), (test_data, test_labels) = get_train_test_data(rgb=True, verbose=False)
    mask_miss, mask_label = get_data_masks(real=test_labels, predictions=predictions, label=EXPECTED_LABEL,
                                           verbose=False)
    # Get the mask for the misclassified items of the selected label
    mask_miss_label = mask_miss[mask_label]
    train_data_gs, test_data_gs = (
        tf.image.rgb_to_grayscale(train_data).numpy(),
        tf.image.rgb_to_grayscale(test_data).numpy()
    )

    # Get the most popular configurations
    df = sample_most_popular(df, group_by='explainer').set_index('explainer')
    print('Exporting the clusters sample images ...')
    explainers = df.index.unique()
    show_progress(0, len(explainers))
    for idx, explainer in enumerate(explainers):
        # Get the best configuration for the explainer
        pick_config = df.loc[explainer]
        clusters, projections, contributions = pick_config[[
            'clusters',
            'projections',
            'contributions'
        ]]
        # Convert the clusters to membership list
        clusters_membership = np.array(Clustering().from_cluster_list(clusters).to_membership_list())
        # Get the mask for the clusters containing misclassified elements of the selected label
        mask_contains_miss_label = np.isin(clusters_membership, np.unique(clusters_membership[mask_miss_label]))

        # Sample some clusters labels containing misclassified items
        clusters_labels = np.unique(clusters_membership[mask_contains_miss_label])
        sample_labels = np.random.choice(clusters_labels, min(4, len(clusters_labels)), replace=False)
        sample_mask = np.isin(clusters_membership, sample_labels)
        # Sort the clusters if a sorting parameter is provided
        if CLUSTERS_SORT_METRIC is not None:
            clusters = sorted_clusters(clusters, metric=CLUSTERS_SORT_METRIC)
            clusters_membership = np.array(Clustering().from_cluster_list(clusters).to_membership_list())
        # Show some correctly classified images for clusters containing also misclassified images
        fig, _ = visualize_clusters_images(
            clusters=clusters_membership[mask_contains_miss_label & ~mask_miss_label & sample_mask],
            images=test_data_gs[mask_label][mask_contains_miss_label & ~mask_miss_label & sample_mask],
            predictions=predictions[mask_label][mask_contains_miss_label & ~mask_miss_label & sample_mask],
            overlay=contributions[mask_contains_miss_label & ~mask_miss_label & sample_mask],
            max_samples=MAX_SAMPLES,
            max_labels=MAX_LABELS,
            cmap='gray_r'
        )
        save_figure(fig, f'out/low_level/{explainer}/clusters_correct_images')
        # Show some incorrectly classified images for clusters containing also misclassified images
        fig, _ = visualize_clusters_images(
            clusters=clusters_membership[mask_contains_miss_label & mask_miss_label & sample_mask],
            images=test_data_gs[mask_label][mask_contains_miss_label & mask_miss_label & sample_mask],
            predictions=predictions[mask_label][mask_contains_miss_label & mask_miss_label & sample_mask],
            overlay=contributions[mask_contains_miss_label & mask_miss_label & sample_mask],
            max_samples=MAX_SAMPLES,
            max_labels=MAX_LABELS,
            cmap='gray_r'
        )
        save_figure(fig, f'out/low_level/{explainer}/clusters_misclassified_images')

        show_progress(idx, len(explainers))
    print()


def heatmaps_silhouette_by_perplexity():
    # Read the data
    df = pd.read_pickle(HEATMAPS_DATA_RAW)

    print('Showing the distribution of the silhouette score by perplexity for the low-level approaches ...')
    # Iterate through the explainers
    explainers = df['explainer'].unique()
    show_progress(0, len(explainers))
    for idx, explainer in enumerate(explainers):
        # Filter the dataframe for the explainer
        explainer_df = df[df['explainer'] == explainer]
        # Show the distribution of the silhouette by perplexity
        fig = plt.figure(figsize=(16, 9))
        sns.boxplot(x='perplexity', y='silhouette', data=explainer_df, color='gray').set_title(
            f'{explainer} silhouette score by perplexity'
        )
        save_figure(fig, f'out/low_level/{explainer}/silhouette_by_perplexity')

        show_progress(idx, len(explainers))
    print()
