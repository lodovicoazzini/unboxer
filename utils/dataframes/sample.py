import pandas as pd

from utils.dataframes.extractor import get_average_popularity_score
from utils.stats import weight_value


def sample_most_popular(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sample the data based on the average popularity of the cluster configurations.
    For each approach, keep the most popular one
    :param df: The dataframe to sample
    :return: The sampled dataframe
    """
    df_sampled = pd.DataFrame.copy(df, deep=True)
    # Find the average popularity score for the configurations
    df_sampled['popularity_score'] = get_average_popularity_score(df_sampled)
    # For each approach, choose configuration with the highest popularity score
    df_sampled['rank'] = df_sampled.groupby('approach')['popularity_score'].rank(
        method='dense',
        ascending=False
    ).astype(int)
    df_sampled = df_sampled[df_sampled['rank'] == 1].drop(columns='rank')
    df_sampled = df_sampled.groupby('approach').first()
    return df_sampled


def sample_highest_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sample the data based on the score for the cluster configurations.
    The score for the cluster configurations is weighted to the number of iterations with non-none score
    :param df: The dataframe to sample
    :return: The sampled dataframe
    """
    df_sampled = pd.DataFrame.copy(df, deep=True)
    # Find the average score and the count of not none entries for each approach and perplexity
    df_sampled['techs'] = df_sampled['dimensionality_reduction_techniques'].apply(
        lambda techs: tuple(str(tech) for tech in techs)
    )
    grouped = df_sampled.groupby(
        ['approach', 'techs']
    )['score'].agg(val='mean', not_none='count')
    # Find the weighted score by the count of not none scores
    max_not_none = grouped['not_none'].max()
    grouped['weighted_val'] = grouped.apply(
        lambda row: weight_value(value=row['val'], weight=row['not_none'], max_weight=max_not_none),
        axis=1
    )
    grouped = grouped.reset_index(level='techs', drop=False)
    # Find the configuration with the highest weighted score
    grouped['rank'] = grouped.groupby('approach')['weighted_val'].rank('dense', ascending=False)
    df_sampled = pd.merge(df_sampled, grouped, on=['approach', 'techs'])
    df_sampled = df_sampled.drop(columns='techs')
    df_sampled = df_sampled[df_sampled['rank'] == 1]
    df_sampled = df_sampled.groupby('approach').first().reset_index(drop=False)
    return df_sampled
