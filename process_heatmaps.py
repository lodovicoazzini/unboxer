import warnings

import numpy as np
import tensorflow as tf
from clusim.clustering import Clustering
from keras.utils.np_utils import to_categorical

from config import HEATMAPS_PROCESS_MODE, EXPLAINERS, DIM_RED_TECHS, CLUS_TECH, ITERATIONS, CLUS_SIM
from config_dirs import CLASSIFIER_PATH
from utils.cluster.compare import compare_approaches
from utils.cluster.preprocessing import distance_matrix
from utils.dataset import get_train_test_data, get_data_masks
from utils.general import save_figure

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
            dim_red_techs=DIM_RED_TECHS,
            clus_tech=CLUS_TECH
        )
        for explainer in EXPLAINERS
    ]
    # Collect the data for the approaches
    df = compare_approaches(
        approaches=approaches,
        data=test_data,
        predictions=predictions_cat,
        iterations=ITERATIONS,
        verbose=True
    )

    print('Computing the distance matrix ...')
    # Compute the distance matrix
    # Remove the configurations with only one cluster
    filtered_df = df[df['num_clusters'] > 1]
    distances_df, fig, ax = distance_matrix(
        heatmaps=[Clustering().from_cluster_list(clusters) for clusters in filtered_df['clusters']],
        dist_func=lambda lhs, rhs: 1 - CLUS_SIM(lhs, rhs),
        names=filtered_df['explainer'],
        show_map=True
    )
    fig.suptitle('Distance matrix for the considered XAI approaches')
    save_figure(fig, 'out/heatmaps/distance_matrix')
