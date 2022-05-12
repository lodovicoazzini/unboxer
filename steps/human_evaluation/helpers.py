import os.path

import numpy as np
import pandas as pd

from config.config_dirs import FEATUREMAPS_DATA, HEATMAPS_DATA, PREDICTIONS, MERGED_DATA, MERGED_DATA_SAMPLED
from config.config_general import EXPECTED_LABEL
from utils.cluster.postprocessing import get_popularity_score
from utils.dataset import get_train_test_data, get_data_masks


def preprocess_featuremaps_data():
    # Read the featuremaps data
    df = pd.read_pickle(FEATUREMAPS_DATA)
    # Compute the total time
    df['time'] = df['map_time'] + df['features_extraction']
    df = df.drop(columns=['map_time', 'features_extraction'])
    # Get the overall information about the approach
    df['approach'] = df.apply(
        lambda row: f'{row["approach"]}({row["map_size"]})_{row["mode"]}',
        axis=1
    )
    df = df.drop(columns=['map_size', 'mode'])

    return df


def preprocess_heatmaps_data():
    # Read the heatmaps data
    df = pd.read_pickle(HEATMAPS_DATA)
    # Make the naming consistent with the featuremaps
    df = df.rename(columns={'explainer': 'approach'})
    df['time'] = df['time_clustering'] + df['time_contributions']
    df['clustering_mode'] = df.apply(lambda row: f'{row["clustering_technique"]}({row["clustering_mode"]})', axis=1)

    # Keep the column of interest
    df = df[
        ['approach', 'clustering_mode', 'time', 'clusters', 'contributions']]
    df = df.rename(columns={'explainer': 'approach'})

    return df


def preprocess_data():
    # Read all the needed data
    featuremaps_data = preprocess_featuremaps_data()
    heatmaps_data = preprocess_heatmaps_data()
    _, (test_data, test_labels) = get_train_test_data(rgb=True)
    predictions = np.loadtxt(PREDICTIONS)
    mask_miss, mask_label = get_data_masks(test_labels, predictions, label=EXPECTED_LABEL)
    mask_miss_label = mask_miss[mask_label]
    misclassified_idxs = np.argwhere(mask_miss_label)

    # Merge the data for the featuremaps and the heatmaps
    merged = pd.concat([featuremaps_data, heatmaps_data]).reset_index(drop=True)
    # Compute additional information
    merged['num_clusters'] = merged['clusters'].apply(len)
    merged['clusters_sizes'] = merged['clusters'].apply(lambda clusters: [len(cluster) for cluster in clusters])
    merged['frac_misses'] = merged['clusters'].apply(
        lambda clusters: [
            len([entry for entry in cluster if entry in misclassified_idxs]) / len(cluster)
            for cluster in clusters
        ]
    )
    merged['frac_mixed'] = merged['frac_misses'].apply(
        lambda misses: len([entry for entry in misses if 0 < entry < 1]) / len(misses)
    )

    merged.to_pickle(MERGED_DATA)

    return merged


def sample_clusters():
    if os.path.exists(MERGED_DATA):
        df = pd.read_pickle(MERGED_DATA)
    else:
        df = preprocess_data()
    df_groups = df.groupby('approach').agg({'clusters': [sum, len]})
    # Find the popularity score of each cluster configuration
    df['popularity_score'] = df.apply(
        lambda row: get_popularity_score(
            clusters_config=row['clusters'],
            all_clusters=df_groups.loc[row['approach']].loc['clusters', 'sum'],
            sample_size=df_groups.loc[row['approach']].loc['clusters', 'len']
        ),
        axis=1
    )
    # For each approach, choose the cluster configuration with the highest popularity score
    df['rank'] = df.groupby('approach')['popularity_score'].rank(method='dense', ascending=False).astype(int)
    sampled_clusters = df[df['rank'] == 1].drop(columns='rank')
    sampled_clusters = sampled_clusters.groupby('approach').first().reset_index()

    sampled_clusters.to_pickle(MERGED_DATA_SAMPLED)

    return sampled_clusters
