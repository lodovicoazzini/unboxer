from collections import Counter

import numpy as np
import pandas as pd


def __get_approach_average_popularity_score(row: pd.Series, groups: pd.DataFrame) -> float:
    clusters_configuration = row['clusters']
    all_clusters = groups.loc[row['approach']].loc['clusters', 'sum']
    sample_size = groups.loc[row['approach']].loc['clusters', 'len']
    # Find the number of occurrences of each cluster configuration
    clusters_popularity = Counter(map(tuple, all_clusters))
    # Find the average fraction of occurrences of each cluster in the sample
    average_popularity = np.average([
        clusters_popularity[tuple(cluster)] / sample_size
        for cluster in clusters_configuration
    ])
    return average_popularity


def get_average_popularity_score(df: pd.DataFrame) -> pd.Series:
    """
    Find the average popularity score for the clusters configurations
    :param df: The dataset for the clusters configurations
    :return: The average popularity score for the cluster configurations
    """
    # For each approach, concatenate the clusters and find the size of the sample
    groups = df.groupby('approach').agg({'clusters': [sum, len]})
    # Find the average popularity score of each configuration
    popularity_scores = df.apply(
        lambda row: __get_approach_average_popularity_score(row=row, groups=groups),
        axis=1
    )
    return popularity_scores
