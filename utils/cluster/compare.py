import datetime
import sys
import time

import pandas as pd
from clusim.clustering import Clustering

from utils.cluster.ClusteringMode import ClusteringMode
from utils.general import show_progress


def compare_approaches(
        approaches: list[ClusteringMode],
        data, predictions,
        iterations: int,
        verbose=False
) -> pd.DataFrame:
    results = []
    # iterate over the approaches in the list
    for idx, approach in enumerate(approaches):
        # get the information about the approach
        explainer = approach.get_explainer()
        clus_tech = approach.get_clustering_technique()
        dim_red_techs = approach.get_dimensionality_reduction_techniques()
        dim_red_techs_names = [dim_red_tech.__class__.__name__ for dim_red_tech in dim_red_techs]
        dim_red_techs_params = [dim_red_tech.get_params() for dim_red_tech in dim_red_techs]

        if verbose:
            perplexity = dim_red_techs_params[-1]['perplexity']
            sys.stdout.write('\r')
            sys.stdout.write(
                f'{idx + 1}/{len(approaches)} - {explainer.__class__.__name__} (perplexity={perplexity})\n'
            )

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

            # append the data
            results.append({
                'clustering_mode': approach.__class__.__name__,
                'explainer': explainer.__class__.__name__,
                'contributions': contributions,
                'clustering_technique': clus_tech.__class__.__name__,
                'dim_red_techs': dim_red_techs_names,
                'dim_red_techs_params': dim_red_techs_params,
                'silhouette': round(score, 3) if score is not None else None,
                'clusters': Clustering().from_membership_list(clusters).to_cluster_list(),
                'projections': projections,
                'num_clusters': len(set(clusters)),
                'time_contributions': round(contributions_time, 5),
                'time_clustering': round(cluster_time, 5),
                'timestamp': datetime.datetime.now()
            })

            # Show the progress
            if verbose:
                show_progress(iteration, iterations)
        print()
    # save the results
    results_df = pd.DataFrame(results)

    # find the optimal configuration
    return results_df
