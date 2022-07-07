import os.path
import warnings
from itertools import product

import numpy as np
import pandas as pd

from config.config_dirs import BEST_CONFIGURATIONS, HEATMAPS_DATA_RAW, \
    HEATMAPS_DATA
from config.config_heatmaps import APPROACH, EXPLAINERS, DIMENSIONALITY_REDUCTION_TECHNIQUES, ITERATIONS
from utils import global_values
from utils.clusters.Approach import OriginalMode, GlobalLatentMode, LocalLatentMode
from utils.clusters.compare import compare_approaches
from utils.dataframes.sample import sample_highest_score

BASE_DIR = f'../out/heatmaps'

APPROACHES = [OriginalMode, LocalLatentMode, GlobalLatentMode]


def get_perplexity(app):
    try:
        return app.get_dimensionality_reduction_techniques()[-1].get_params()['perplexity']
    except IndexError:
        return np.nan


def main():
    # Ignore warnings from tensorflow
    warnings.filterwarnings('ignore')

    # Collect the approaches to use
    print('Collecting the approaches ...')
    # Select the approach from the configurations
    approach = APPROACHES[APPROACH]
    # Select the dimensionality reduction techniques based on the approach
    dimensionality_reduction_techniques = [[]] if approach is OriginalMode else DIMENSIONALITY_REDUCTION_TECHNIQUES
    # If the processing mode is the original one, or there are no best logs -> try all the combinations
    if not os.path.exists(BEST_CONFIGURATIONS) or approach is OriginalMode:
        # Collect the approaches
        approaches = [
            approach(
                explainer=explainer(global_values.classifier),
                dimensionality_reduction_techniques=dimensionality_reduction_technique
            )
            for explainer, dimensionality_reduction_technique
            in product(EXPLAINERS, dimensionality_reduction_techniques)
        ]
    # The processing mode is not original and there is a log -> read the configurations from the log
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
                    approach(
                        explainer=explainer(global_values.classifier),
                        dimensionality_reduction_techniques=best_config['dimensionality_reduction_techniques']
                    )
                )
            except KeyError:
                # Collect the approaches
                for approach in [
                    approach(
                        explainer=explainer(global_values.classifier),
                        dimensionality_reduction_techniques=dimensionality_reduction_technique
                    )
                    for explainer, dimensionality_reduction_technique
                    in product(explainer, dimensionality_reduction_techniques)
                ]:
                    approaches.append(approach)

    # Collect the data for the approaches
    print('Collecting the data for the approaches ...')
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
