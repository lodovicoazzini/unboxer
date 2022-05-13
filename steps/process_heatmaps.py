import os.path
import warnings
from itertools import product

import pandas as pd

from config.config_dirs import BEST_CONFIGURATIONS, HEATMAPS_DATA_RAW, \
    HEATMAPS_DATA
from config.config_heatmaps import HEATMAPS_PROCESS_MODE, EXPLAINERS, DIMENSIONALITY_REDUCTION_TECHNIQUES, \
    CLUSTERING_TECHNIQUE, ITERATIONS
from utils import globals
from utils.cluster.compare import compare_approaches
from utils.cluster.sample import sample_highest_score

BASE_DIR = f'../out/heatmaps'


def main():
    warnings.filterwarnings('ignore')

    # Collect the approaches to use
    print('Collecting the approaches ...')
    if not os.path.exists(BEST_CONFIGURATIONS):
        approaches = [
            HEATMAPS_PROCESS_MODE(
                mask=globals.mask_label,
                explainer=explainer(globals.classifier),
                dim_red_techs=dimensionality_reduction_technique,
                clus_tech=CLUSTERING_TECHNIQUE
            )
            for explainer, dimensionality_reduction_technique
            in product(EXPLAINERS, DIMENSIONALITY_REDUCTION_TECHNIQUES)
        ]
    else:
        # Read the data about the best configurations
        best_configurations = pd.read_csv(BEST_CONFIGURATIONS).set_index('approach')
        # Find the best settings for each approach
        approaches = []
        for explainer in EXPLAINERS:
            try:
                # Find the best configuration to use
                best_config = best_configurations.loc[explainer(globals.classifier).__class__.__name__]
                approaches.append(
                    HEATMAPS_PROCESS_MODE(
                        mask=globals.mask_label,
                        explainer=explainer(globals.classifier),
                        dim_red_techs=best_config['dimensionality_reduction_techniques'],
                        clus_tech=CLUSTERING_TECHNIQUE
                    )
                )
            except KeyError:
                # No best configuration for the explainer
                for approach in [
                    HEATMAPS_PROCESS_MODE(
                        mask=globals.mask_label,
                        explainer=explainer(globals.classifier),
                        dim_red_techs=dim_red_techs,
                        clus_tech=CLUSTERING_TECHNIQUE
                    )
                    for explainer, dim_red_techs in product([explainer], DIMENSIONALITY_REDUCTION_TECHNIQUES)
                ]:
                    approaches.append(approach)

    # Collect the data for the approaches
    print('Collecting the data for the approaches ...')
    df_raw = compare_approaches(
        approaches=approaches,
        iterations=ITERATIONS,
        get_info=lambda app: app.get_dimensionality_reduction_techniques()[-1].get_params()['perplexity'],
        verbose=True
    )
    # Export the raw data
    df_raw.to_pickle(HEATMAPS_DATA_RAW)

    # Find the best configuration for each explainer
    df_sampled = sample_highest_score(df_raw)
    best_configurations_tuples = list(
        df_sampled.reset_index()[['approach', 'dimensionality_reduction_techniques']].itertuples(index=False, name=None)
    )
    # Filter the dataset for the entries corresponding to the best configuration
    df_sampled = df_raw[df_raw[
        ['approach', 'dimensionality_reduction_techniques']].apply(tuple, axis=1).isin(best_configurations_tuples)
    ]
    # Remove the rows where the score is none -> black hole cluster
    df_sampled = df_sampled.dropna(subset=['score'])
    # Export the sampled data
    df_sampled.to_pickle(HEATMAPS_DATA)

    # Export the best configurations
    best_configs_df = df_sampled[
        ['approach', 'dimensionality_reduction_techniques']
    ].groupby('approach').first().reset_index(drop=False)
    best_configs_df.to_csv(BEST_CONFIGURATIONS, index=False)

    return df_sampled


if __name__ == '__main__':
    main()
