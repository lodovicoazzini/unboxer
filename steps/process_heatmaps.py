import os.path
import warnings
from itertools import product

import numpy as np
import pandas as pd

from config.config_dirs import BEST_CONFIGURATIONS, HEATMAPS_DATA_RAW, \
    HEATMAPS_DATA
from config.config_heatmaps import HEATMAPS_PROCESS_MODE, EXPLAINERS, DIMENSIONALITY_REDUCTION_TECHNIQUES, \
    CLUSTERING_TECHNIQUE, ITERATIONS
from utils import global_values
from utils.clusters.ClusteringMode import OriginalMode
from utils.clusters.compare import compare_approaches
from utils.dataframes.sample import sample_highest_score

BASE_DIR = f'../out/heatmaps'


def main():
    warnings.filterwarnings('ignore')

    # Collect the approaches to use
    print('Collecting the approaches ...')
    if not os.path.exists(BEST_CONFIGURATIONS):
        # No best configuration for the explainer
        if HEATMAPS_PROCESS_MODE is OriginalMode:
            approaches = [
                HEATMAPS_PROCESS_MODE(
                    explainer=explainer(global_values.classifier),
                    dimensionality_reduction_techniques=[],
                    clustering_technique=CLUSTERING_TECHNIQUE
                )
                for explainer in EXPLAINERS
            ]
        else:
            approaches = [
                HEATMAPS_PROCESS_MODE(
                    explainer=explainer(global_values.classifier),
                    dimensionality_reduction_techniques=dimensionality_reduction_technique,
                    clustering_technique=CLUSTERING_TECHNIQUE
                )
                for explainer, dimensionality_reduction_technique
                in product(EXPLAINERS, DIMENSIONALITY_REDUCTION_TECHNIQUES)
            ]
    else:
        # Read the data about the best configurations
        best_configurations = pd.read_pickle(BEST_CONFIGURATIONS).set_index('approach')
        # Find the best settings for each approach
        approaches = []
        for explainer in EXPLAINERS:
            try:
                # Find the best configuration to use
                best_config = best_configurations.loc[explainer(global_values.classifier).__class__.__name__]
                approaches.append(
                    HEATMAPS_PROCESS_MODE(
                        explainer=explainer(global_values.classifier),
                        dimensionality_reduction_techniques=best_config['dimensionality_reduction_techniques'],
                        clustering_technique=CLUSTERING_TECHNIQUE
                    )
                )
            except KeyError:
                # No best configuration for the explainer
                if HEATMAPS_PROCESS_MODE is OriginalMode:
                    approaches.append(
                        HEATMAPS_PROCESS_MODE(
                            explainer=explainer(global_values.classifier),
                            dimensionality_reduction_techniques=[],
                            clustering_technique=CLUSTERING_TECHNIQUE
                        )
                    )
                else:
                    for approach in [
                        HEATMAPS_PROCESS_MODE(
                            explainer=explainer(global_values.classifier),
                            dimensionality_reduction_techniques=dim_red_techs,
                            clustering_technique=CLUSTERING_TECHNIQUE
                        )
                        for explainer, dim_red_techs in product([explainer], DIMENSIONALITY_REDUCTION_TECHNIQUES)
                    ]:
                        approaches.append(approach)

    # Collect the data for the approaches
    print('Collecting the data for the approaches ...')

    def get_perplexity(app):
        try:
            return app.get_dimensionality_reduction_techniques()[-1].get_params()['perplexity']
        except IndexError:
            return np.nan

    df_raw = compare_approaches(
        approaches=approaches,
        iterations=ITERATIONS,
        get_info=lambda app: f"perplexity: {get_perplexity(app)}" if get_perplexity(app) != np.nan else "Original Mode"
    )
    # Export the raw data
    df_raw.to_pickle(HEATMAPS_DATA_RAW)

    # Find the best configuration for each explainer
    df_sampled = sample_highest_score(df_raw)
    best_configurations_tuples = list(
        df_sampled[['approach', 'dimensionality_reduction_techniques']].itertuples(index=False, name=None)
    )
    # Filter the dataset for the entries corresponding to the best configuration
    df_sampled = df_raw[df_raw[
        ['approach', 'dimensionality_reduction_techniques']].apply(tuple, axis=1).isin(best_configurations_tuples)
    ]
    # Remove the rows where the score is none -> black hole clusters
    df_sampled = df_sampled.dropna(subset=['score'])
    # Export the sampled data
    df_sampled.to_pickle(HEATMAPS_DATA)

    # Export the best configurations
    best_configs_df = df_sampled[
        ['approach', 'dimensionality_reduction_techniques']
    ].groupby('approach').first().reset_index(drop=False)
    best_configs_df.to_pickle(BEST_CONFIGURATIONS)

    return df_sampled


if __name__ == '__main__':
    main()
