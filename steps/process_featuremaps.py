import os.path
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from clusim.clustering import Clustering
from clusim.sim import element_sim
from keras.utils.np_utils import to_categorical
from sklearn.manifold import TSNE

import feature_map.mnist.feature_map as feature_map_generator
from config.config_dirs import MODEL, PREDICTIONS, FEATUREMAPS_DATA_RAW, FEATUREMAPS_DATA
from config.config_featuremaps import FEATUREMAPS_CLUSTERS_MODE
from config.config_general import CLUSTERS_SORT_METRIC, MAX_SAMPLES, MAX_LABELS
from utils.cluster.postprocessing import sorted_clusters
from utils.cluster.preprocessing import extract_maps_clusters, distance_matrix
from utils.cluster.visualize import visualize_clusters_projections, visualize_clusters_images
from utils.dataset import get_train_test_data, get_data_masks
from utils.general import save_figure

BASE_DIR = f'out/featuremaps/{FEATUREMAPS_CLUSTERS_MODE.name}'


def main():
    warnings.filterwarnings('ignore')

    # Import the featuremaps data or generate it if not there
    if os.path.exists(FEATUREMAPS_DATA_RAW):
        featuremaps_df = pd.read_pickle(FEATUREMAPS_DATA_RAW)
    else:
        featuremaps_df = feature_map_generator.main()

    # Process the feature-maps and get the dataframe
    print('Extracting the clusters data from the feature-maps ...')
    featuremaps_df = extract_maps_clusters(featuremaps_df)
    featuremaps_df.to_pickle(FEATUREMAPS_DATA)

    # Compute the distance matrix
    print('Computing the distance matrix ...')
    filtered_featuremaps_df = featuremaps_df[featuremaps_df['mode'] == FEATUREMAPS_CLUSTERS_MODE.value]
    featuremaps_clusters = [
        Clustering().from_cluster_list(clusters_configuration)
        for clusters_configuration in filtered_featuremaps_df['clusters'].values
    ]
    filtered_featuremaps_df['complete_approach'] = filtered_featuremaps_df.apply(
        lambda row: f'{row["approach"]}({row["map_size"]})_{row["mode"]}',
        axis=1
    )
    dist_matrix, fig, ax = distance_matrix(
        featuremaps_clusters,
        lambda l, r: 1 - element_sim(l, r),
        show_map=True,
        names=filtered_featuremaps_df['complete_approach']
    )
    ax.set_title('Distance matrix for the feature combinations')
    save_figure(fig, f'{BASE_DIR}/distance_matrix')
    print()

    print('Getting ready to show the clusters projections ...')
    # Get the train and test data
    (train_data, train_labels), (test_data, test_labels) = get_train_test_data(rgb=True)
    train_labels_cat, test_labels_cat = to_categorical(train_labels), to_categorical(test_labels)
    train_data_gs, test_data_gs = (
        tf.image.rgb_to_grayscale(train_data).numpy(),
        tf.image.rgb_to_grayscale(test_data).numpy()
    )
    # Load the model
    classifier = tf.keras.models.load_model(MODEL)
    # Load the predictions
    predictions = np.loadtxt(PREDICTIONS)
    predictions_cat = to_categorical(predictions, num_classes=len(set(train_labels)))
    # Get the masks to filter the data
    mask_miss, mask_label = get_data_masks(test_labels, predictions, label=5)
    mask_miss_label = mask_miss[mask_label]
    # Project the data in the 2d latent space
    projections = TSNE(perplexity=3).fit_transform(
        test_data_gs[mask_label].reshape(test_data_gs[mask_label].shape[0], -1)
    )

    print('Showing the clusters projections and some sample images ...')
    # Show the clusters projections for each feature combination
    for feature_combination, clusters in zip(filtered_featuremaps_df['complete_approach'].values, featuremaps_clusters):
        clusters_membership = np.array(clusters.to_membership_list())
        # Get the mask for the clusters containing misclassified elements of the selected label
        mask_contains_miss_label = np.isin(clusters_membership, np.unique(clusters_membership[mask_miss_label]))
        # Show the clusters projections
        fig, ax = visualize_clusters_projections(
            projections=projections,
            clusters=clusters_membership,
            mask=mask_miss_label
        )
        fig.suptitle(f'Clusters projections for the features {feature_combination}')
        save_figure(fig, f'{BASE_DIR}/clusters_projections_{feature_combination}')

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

    return featuremaps_df


if __name__ == '__main__':
    main()
