import warnings

import numpy as np
import tensorflow as tf
from clusim.clustering import Clustering
from clusim.sim import element_sim
from keras.utils.np_utils import to_categorical
from sklearn.manifold import TSNE

from config import CLASSIFIER_PATH, PREDICTIONS_PATH, FEATUREMAPS_CLUSTERS_MODE
from utils.cluster.preprocessing import extract_maps_clusters, distance_matrix
from utils.cluster.visualize import visualize_clusters_projections, visualize_clusters_images
from utils.dataset import get_train_test_data, get_data_masks
from utils.general import save_figure

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # Process the feature-maps and get the dataframe
    print('Extracting the clusters data from the feature-maps ...')
    featuremaps_df = extract_maps_clusters()

    # Compute the distance matrix
    print('Computing the distance matrix ...')
    featuremaps_clusters = [
        Clustering().from_cluster_list(clusters_configuration)
        for clusters_configuration in featuremaps_df[FEATUREMAPS_CLUSTERS_MODE.value].values
    ]
    distance_matrix, fig, ax = distance_matrix(
        featuremaps_clusters,
        lambda l, r: 1 - element_sim(l, r),
        show_map=True,
        names=featuremaps_df.index
    )
    ax.set_title('Distance matrix for the feature combinations')
    save_figure(fig, f'out/feature_maps/distance_matrix')
    print(distance_matrix)
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
    classifier = tf.keras.models.load_model(CLASSIFIER_PATH)
    # Load the predictions
    predictions = np.loadtxt(PREDICTIONS_PATH)
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
    for feature_combination, clusters in zip(featuremaps_df.index, featuremaps_clusters):
        clusters = np.array(clusters.to_membership_list())
        # Get the mask for the clusters containing misclassified elements of the selected label
        mask_contains_miss_label = np.isin(clusters, np.unique(clusters[mask_miss_label]))
        # Show the correct classifications
        fig, ax = visualize_clusters_projections(
            projections=projections[~mask_miss_label & mask_contains_miss_label],
            clusters=clusters[~mask_miss_label & mask_contains_miss_label],
            cmap='tab10', marker='.'
        )
        # Show the misclassified items
        visualize_clusters_projections(
            projections=projections[mask_miss_label],
            clusters=clusters[mask_miss_label],
            fig=fig, ax=ax, cmap='tab10', marker='X', label_prefix='mis'
        )
        fig.suptitle(f'Clusters projections for the features {feature_combination}')
        save_figure(fig, f'out/feature_maps/clusters_projections_{feature_combination}')

        # sample some clusters labels containing misclassified items
        sample_labels = np.random.choice(np.unique(clusters[mask_contains_miss_label]), 4, replace=False)
        sample_mask = np.isin(clusters, sample_labels)
        # show some correctly classified images for clusters containing also misclassified images
        fig, _ = visualize_clusters_images(
            clusters=clusters[mask_contains_miss_label & ~mask_miss_label & sample_mask],
            images=test_data_gs[mask_label][mask_contains_miss_label & ~mask_miss_label & sample_mask],
            predictions=predictions[mask_label][mask_contains_miss_label & ~mask_miss_label & sample_mask],
            max_labels=4, max_samples=3,
            cmap='gray_r'
        )
        fig.suptitle(f'Correctly classified items for {feature_combination}')
        save_figure(fig, f'out/feature_maps/correct_samples_{feature_combination}')
        # show some incorrectly classified images for clusters containing also misclassified images
        fig, _ = visualize_clusters_images(
            clusters=clusters[mask_contains_miss_label & mask_miss_label & sample_mask],
            images=test_data_gs[mask_label][mask_contains_miss_label & mask_miss_label & sample_mask],
            predictions=predictions[mask_label][mask_contains_miss_label & mask_miss_label & sample_mask],
            max_labels=4, max_samples=3,
            cmap='gray_r'
        )
        fig.suptitle(f'Misclassified classified items for {feature_combination}')
        save_figure(fig, f'out/feature_maps/misclassified_samples_{feature_combination}')
