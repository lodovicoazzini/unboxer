import math

import pandas as pd

from utils.cluster.postprocessing import get_popularity_score


def sample_most_popular(df):
    selected_clusters = pd.DataFrame.copy(df, deep=True)
    # Find all the clusters configurations and the number of samples for each approach
    approaches_merged = df.groupby('approach').agg({'clusters': [sum, len]})
    # Find the popularity score of each cluster configuration
    selected_clusters['popularity_score'] = selected_clusters.apply(
        lambda row: get_popularity_score(
            row['clusters'],
            approaches_merged.loc[row['approach']].loc['clusters', 'sum'],
            approaches_merged.loc[row['approach']].loc['clusters', 'len']
        ),
        axis=1
    )
    # For each approach, choose the cluster configuration with the highest popularity score
    selected_clusters['rank'] = selected_clusters.groupby('approach')['popularity_score'].rank(
        method='dense',
        ascending=False
    ).astype(int)
    selected_clusters = selected_clusters[selected_clusters['rank'] == 1].drop(columns='rank')
    # Make sure to have only one row for each group
    selected_clusters = selected_clusters.groupby('approach').first()

    return selected_clusters


def sample_highest_score(df: pd.DataFrame) -> pd.DataFrame:
    # Find the average score and the count of not none entries for each approach and perplexity
    grouped = df.groupby(['approach', 'dimensionality_reduction_technique'])['score'].agg(val='mean', not_none='count')
    # Find the weighted score by the count of not none scores
    max_not_none = grouped['not_none'].max()
    grouped['weighted_val'] = grouped.apply(
        lambda row: row['val'] * math.log((math.e - 1) * row['not_none'] / max_not_none + 1),
        axis=1
    ).reset_index(level='dimensionality_reduction_technique', drop=False)
    # Find the configuration with the highest weighted score
    grouped['rank'] = grouped.groupby('approach')['weighted_val'].rank('dense', ascending=False)
    best_configs_df = grouped[grouped['rank'] == 1]
    best_configs_df = best_configs_df.groupby('approach').first()

    return best_configs_df
