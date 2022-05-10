import pandas as pd

from utils.cluster.postprocessing import get_popularity_score


def sample_most_popular(df, group_by):
    selected_clusters = pd.DataFrame.copy(df, deep=True)
    # Find all the clusters configurations and the number of samples for each approach
    approaches_merged = df.groupby(group_by).agg({'clusters': [sum, len]})
    # Find the popularity score of each cluster configuration
    selected_clusters['popularity_score'] = selected_clusters.apply(
        lambda row: get_popularity_score(
            row['clusters'],
            approaches_merged.loc[row[group_by]].loc['clusters', 'sum'],
            approaches_merged.loc[row[group_by]].loc['clusters', 'len']
        ),
        axis=1
    )
    # For each approach, choose the cluster configuration with the highest popularity score
    selected_clusters['rank'] = selected_clusters.groupby(group_by)['popularity_score'].rank(
        method='dense',
        ascending=False
    ).astype(int)
    selected_clusters = selected_clusters[selected_clusters['rank'] == 1].drop(columns='rank')
    # Make sure to have only one row for each group
    selected_clusters = selected_clusters.groupby(group_by).first().reset_index()

    return selected_clusters
