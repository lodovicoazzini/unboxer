import warnings
from itertools import product

import numpy as np
import seaborn as sns
import tensorflow as tf
from clusim.clustering import Clustering
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt

from config import HEATMAPS_PROCESS_MODE, EXPLAINERS, DIM_RED_TECHS, CLUS_TECH, ITERATIONS, CLUS_SIM
from config_dirs import CLASSIFIER_PATH
from utils.cluster.compare import compare_approaches
from utils.cluster.preprocessing import distance_matrix
from utils.cluster.visualize import visualize_clusters_projections, visualize_clusters_images
from utils.dataset import get_train_test_data, get_data_masks
from utils.general import weight_not_null, save_figure

BASE_DIR = f'out/heatmaps'

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # Get the data
    print('Importing the data ...')
    (train_data, train_labels), (test_data, test_labels) = get_train_test_data(rgb=True, verbose=True)
    train_labels_cat, test_labels_cat = to_categorical(train_labels), to_categorical(test_labels)
    train_data_gs, test_data_gs = (
        tf.image.rgb_to_grayscale(train_data).numpy(),
        tf.image.rgb_to_grayscale(test_data).numpy()
    )
    # Load the model and the predictions
    print('Loading the model and the predictions ...')
    classifier = tf.keras.models.load_model(CLASSIFIER_PATH)
    predictions = np.loadtxt('in/predictions.csv')
    predictions_cat = to_categorical(predictions, num_classes=len(set(train_labels)))
    # Get the masks to filter the data
    mask_miss, mask_label = get_data_masks(test_labels, predictions, label=5, verbose=True)

    print('Collecting the heatmaps data ...')
    # Compare the approaches in the config file
    approaches = [
        HEATMAPS_PROCESS_MODE(
            mask=mask_label,
            explainer=explainer(classifier),
            dim_red_techs=dim_red_techs,
            clus_tech=CLUS_TECH
        )
        for explainer, dim_red_techs in product(EXPLAINERS, DIM_RED_TECHS)
    ]
    # Collect the data for the approaches
    df = compare_approaches(
        approaches=approaches,
        data=test_data,
        predictions=predictions_cat,
        iterations=ITERATIONS,
        verbose=True
    )

    # Find the best configuration for each explainer
    df['perplexity'] = df['dim_red_techs_params'].apply(lambda params: float(params[-1]['perplexity']))
    weighted_df = weight_not_null(
        df,
        group_by=['explainer', 'perplexity'],
        agg_column='silhouette'
    ).reset_index(level='perplexity', drop=False)
    weighted_df['rank'] = weighted_df.groupby('explainer')['weighted_val'].rank('dense', ascending=False)
    best_configs_df = weighted_df[weighted_df['rank'] == 1]
    best_config_combs = list(
        best_configs_df.reset_index()[['explainer', 'perplexity']].itertuples(index=False, name=None)
    )
    # Filter the dataset for the entries corresponding to the best configuration for each explainer
    filtered_df = df[df[['explainer', 'perplexity']].apply(tuple, axis=1).isin(best_config_combs)]
    filtered_df.to_csv('logs/heatmaps_data.csv', index=False)

    print('Computing the distance matrix ...')
    # Get the data for the best configurations
    # Compute the distance matrix
    # Remove the configurations with only one cluster
    distance_matrix_df = filtered_df[filtered_df['num_clusters'] > 1]
    distances_df, fig, ax = distance_matrix(
        heatmaps=[Clustering().from_cluster_list(clusters) for clusters in distance_matrix_df['clusters']],
        dist_func=lambda lhs, rhs: 1 - CLUS_SIM(lhs, rhs),
        names=distance_matrix_df['explainer'],
        show_map=True
    )
    fig.suptitle('Distance matrix for the considered XAI approaches')
    save_figure(fig, f'{BASE_DIR}/distance_matrix')

    print('Showing images for all the explainers')
    # Iterate through the explainers
    for explainer in df['explainer'].unique():
        # Filter the dataframe for the explainer
        explainer_df = df[df['explainer'] == explainer]

        # Show the distribution of the silhouette by perplexity
        fig = plt.figure(figsize=(16, 9))
        sns.boxplot(x='perplexity', y='silhouette', data=explainer_df, color='gray').set_title(
            f'{explainer} silhouette score by perplexity'
        )
        save_figure(fig, f'{BASE_DIR}/{explainer}/silhouette_by_perplexity')

        # Get the data for the best configuration of the explainer
        explainer_best_df = filtered_df[filtered_df['explainer'] == explainer]
        # Get the best configuration for the explainer
        pick_config = explainer_best_df.sort_values('silhouette', ascending=False).iloc[0]
        clusters, score, projections, contributions = pick_config[[
            'clusters',
            'silhouette',
            'projections',
            'contributions'
        ]]
        # Convert the clusters to membership list
        clusters = np.array(Clustering().from_cluster_list(clusters).to_membership_list())
        # Get the mask for the misclassified items of the selected label
        mask_miss_label = mask_miss[mask_label]
        # Get the mask for the clusters containing misclassified elements of the selected label
        mask_contains_miss_label = np.isin(clusters, np.unique(clusters[mask_miss_label]))

        # Visualize the projections of the contributions for the correct predictions
        fig, ax = visualize_clusters_projections(
            projections=projections[~mask_miss_label & mask_contains_miss_label],
            clusters=clusters[~mask_miss_label & mask_contains_miss_label],
            cmap='tab10', marker='.'
        )
        # Visualize the projections of the contributions for the wrong predictions
        visualize_clusters_projections(
            projections=projections[mask_miss_label],
            clusters=clusters[mask_miss_label],
            fig=fig, ax=ax, cmap='tab10', marker='X', label_prefix='mis'
        )
        ax.set_title(f'{explainer} clusters projections')
        save_figure(fig, f'{BASE_DIR}/{explainer}/clusters_projections')

        # Sample some clusters labels containing misclassified items
        clusters_labels = np.unique(clusters[mask_contains_miss_label])
        sample_labels = np.random.choice(clusters_labels, min(4, len(clusters_labels)), replace=False)
        sample_mask = np.isin(clusters, sample_labels)
        # Show some correctly classified images for clusters containing also misclassified images
        fig, _ = visualize_clusters_images(
            clusters=clusters[mask_contains_miss_label & ~mask_miss_label & sample_mask],
            images=test_data_gs[mask_label][mask_contains_miss_label & ~mask_miss_label & sample_mask],
            predictions=predictions[mask_label][mask_contains_miss_label & ~mask_miss_label & sample_mask],
            overlay=contributions[mask_contains_miss_label & ~mask_miss_label & sample_mask],
            max_samples=3,
            cmap='gray_r'
        )
        save_figure(fig, f'{BASE_DIR}/{explainer}/clusters_correct_images')
        # Show some incorrectly classified images for clusters containing also misclassified images
        fig, _ = visualize_clusters_images(
            clusters=clusters[mask_contains_miss_label & mask_miss_label & sample_mask],
            images=test_data_gs[mask_label][mask_contains_miss_label & mask_miss_label & sample_mask],
            predictions=predictions[mask_label][mask_contains_miss_label & mask_miss_label & sample_mask],
            overlay=contributions[mask_contains_miss_label & mask_miss_label & sample_mask],
            max_samples=3,
            cmap='gray_r'
        )
        save_figure(fig, f'{BASE_DIR}/{explainer}/clusters_misclassified_images')
