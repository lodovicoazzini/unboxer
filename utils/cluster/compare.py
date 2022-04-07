import datetime
import time

import pandas as pd

from utils.cluster.ClusteringMode import ClusteringMode


def compare_approaches(
        approaches: list[ClusteringMode],
        data, predictions,
        iterations: int,
        verbose=False
) -> pd.DataFrame:
    results = []
    # iterate over the approaches in the list
    for approach in approaches:
        # get the information about the approach
        explainer = approach.get_explainer()
        clus_tech = approach.get_clustering_technique()
        dim_red_techs = approach.get_dimensionality_reduction_techniques()
        dim_red_techs_names = [dim_red_tech.__class__.__name__ for dim_red_tech in dim_red_techs]
        dim_red_techs_params = [dim_red_tech.get_params() for dim_red_tech in dim_red_techs]
        if verbose:
            print(f"""
APPROACH: {approach.__class__.__name__}
EXPLAINER: {explainer.__class__.__name__}
CLUS TECH: {clus_tech.__class__.__name__}
DIM RED TECH: {dim_red_techs_names}
DIM RED TECH PARAMS: {dim_red_techs_params}
            """)
        # generate the contributions
        start = time.time()
        contributions = approach.generate_contributions(data, predictions)
        contributions_time = time.time() - start
        # repeat the process
        for iteration in range(iterations):
            # cluster the contributions
            start = time.time()
            try:
                clusters, score, projections = approach.cluster_contributions(contributions)
            except ValueError:
                # TODO think about how to handle the no cluster situation
                # only one cluster generated -> silhouette error
                clusters, score, projections = [], None, []
            cluster_time = time.time() - start
            if verbose:
                print(f'{iteration + 1}/{iterations} -> {score if score is not None else "NO CLUSTERS"}')

            # append the data
            results.append({
                'CLUSTERING_MODE': approach.__class__.__name__,
                'EXPLAINER': explainer.__class__.__name__,
                'CLUSTERING_TECHNIQUE': clus_tech.__class__.__name__,
                'DIMENSIONALITY_REDUCTION_TECHNIQUE': dim_red_techs_names,
                'DIMENSIONALITY_REDUCTION_TECHNIQUE_PARAMS': dim_red_techs_params,
                'SILHOUETTE': round(score, 3),
                'CLUSTERS': clusters,
                'TIME_CONTRIBUTIONS': round(contributions_time, 5),
                'TIME_CLUSTERING': round(cluster_time, 5),
                'TIMESTAMP': datetime.datetime.now()
            })

    # save the results
    results_df = pd.DataFrame(results)

    # find the optimal configuration
    return results_df
