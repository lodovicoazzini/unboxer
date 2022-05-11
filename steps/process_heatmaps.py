import os.path
import warnings
from itertools import product

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.manifold import TSNE

from config.config_dirs import MODEL, BEST_CONFIGURATIONS, PREDICTIONS, HEATMAPS_DATA_RAW, \
    HEATMAPS_DATA
from config.config_execution import HEATMAPS_PROCESS_MODE, EXPLAINERS, DIM_RED_TECHS, CLUS_TECH, ITERATIONS, \
    CHOSEN_LABEL
from steps import model
from utils.cluster.compare import compare_approaches
from utils.dataset import get_train_test_data, get_data_masks
from utils.general import weight_not_null

BASE_DIR = f'../out/heatmaps'


def main():
    warnings.filterwarnings('ignore')

    # Get the data
    print('Importing the data ...')
    (train_data, train_labels), (test_data, test_labels) = get_train_test_data(rgb=True, verbose=True)
    # Load the model
    print('Loading the model ...')
    if os.path.exists(MODEL):
        classifier = tf.keras.models.load_model(MODEL)
    else:
        classifier = model.create_model()
    # Load the predictions
    print('Loading the predictions ...')
    if os.path.exists(PREDICTIONS):
        predictions = np.loadtxt(PREDICTIONS)
    else:
        predictions = model.generate_predictions(classifier=classifier, test_data=test_data)
    # Convert the predictions to categorical
    predictions_cat = to_categorical(predictions, num_classes=len(set(train_labels)))
    # Get the masks to filter the data
    mask_miss, mask_label = get_data_masks(real=test_labels, predictions=predictions, label=CHOSEN_LABEL, verbose=True)

    # Collect the approaches to use
    print('Collecting the approaches to use ...')
    if not os.path.exists(BEST_CONFIGURATIONS):
        # No best configuration stored
        print('Finding the best configuration for each approach')
        approaches = [
            HEATMAPS_PROCESS_MODE(
                mask=mask_label,
                explainer=explainer(classifier),
                dim_red_techs=dim_red_techs,
                clus_tech=CLUS_TECH
            )
            for explainer, dim_red_techs in product(EXPLAINERS, DIM_RED_TECHS)
        ]
    else:
        # Read the data about the best configurations
        best_configs = pd.read_csv(BEST_CONFIGURATIONS)
        best_configs = best_configs.set_index('explainer')
        # Find the best settings for each approach
        approaches = []
        for explainer in EXPLAINERS:
            try:
                # Find the best configuration to use
                best_config = best_configs.loc[explainer(classifier).__class__.__name__]
                approaches.append(
                    HEATMAPS_PROCESS_MODE(
                        mask=mask_label,
                        explainer=explainer(classifier),
                        dim_red_techs=[TSNE(perplexity=best_config['perplexity'])],
                        clus_tech=CLUS_TECH
                    )
                )
            except KeyError:
                # No best configuration for the explainer
                for approach in [
                    HEATMAPS_PROCESS_MODE(
                        mask=mask_label,
                        explainer=explainer(classifier),
                        dim_red_techs=dim_red_techs,
                        clus_tech=CLUS_TECH
                    )
                    for explainer, dim_red_techs in product([explainer], DIM_RED_TECHS)
                ]:
                    approaches.append(approach)

    # Collect the data for the approaches

    print('Collecting the data for the approaches ...')
    df = compare_approaches(
        approaches=approaches,
        data=test_data,
        predictions=predictions_cat,
        iterations=ITERATIONS,
        verbose=True
    )
    # Extract the perplexity from the parameters
    df['perplexity'] = df['dim_red_techs_params'].apply(lambda params: float(params[-1]['perplexity']))
    # Save the overall data
    df.to_pickle(HEATMAPS_DATA_RAW)

    # Find the best configuration for each explainer
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
    filtered_df = filtered_df[~filtered_df['silhouette'].isna()]
    # Save the data for the chosen configurations
    filtered_df.to_pickle(HEATMAPS_DATA)

    # Save the best configurations
    best_configs_df = filtered_df[['explainer', 'perplexity']].groupby('explainer').min().reset_index(drop=False)
    best_configs_df.to_csv(BEST_CONFIGURATIONS, index=False)

    return filtered_df, df


if __name__ == '__main__':
    main()
