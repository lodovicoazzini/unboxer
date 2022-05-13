import time
from typing import Callable

import numpy as np
import pandas as pd
from clusim.clustering import Clustering

from utils import globals
from utils.cluster.ClusteringMode import ClusteringMode
from utils.general import show_progress, echo_progress_message


def compare_approaches(
        approaches: list[ClusteringMode],
        iterations: int,
        get_info: Callable[[ClusteringMode], str] = None,
        verbose: bool = False
) -> pd.DataFrame:
    """
    Compare a list of approaches and return the collected data
    :param approaches: The list of approaches to compare
    :param iterations: The number of iterations to process each approach
    :param get_info: A function to get the string information about the current approach
    :param verbose: Weather to print some information about the run
    :return: A dataframe with the data collected during the tries
    """
    # Iterate over the approaches in the list
    data = []
    for idx, approach in enumerate(approaches):
        # Extract some information about the current approach
        explainer = approach.get_explainer()
        clustering_technique = approach.get_clustering_technique()
        dimensionality_reduction_techniques = approach.get_dimensionality_reduction_techniques()

        if verbose and get_info is not None:
            echo_progress_message(
                iteration=idx,
                iterations=len(approaches),
                message=f'{explainer.__class__.__name__} ({get_info(approach)})'
            )

        # Generate the contributions
        start = time.time()
        contributions = approach.generate_contributions(globals.test_data, globals.predictions_cat)
        contributions_time = time.time() - start

        # Repeat for the number of iterations
        if verbose:
            show_progress(0, iterations)
        for iteration in range(iterations):
            # Cluster the contributions
            start = time.time()
            try:
                clusters, score, projections = approach.cluster_contributions(contributions)
            except ValueError:
                # Black hole cluster -> silhouette error
                clusters, score, projections = [], np.nan, []
            cluster_time = time.time() - start

            # Collect the information for the run
            data.append({
                'approach': explainer.__class__.__name__,
                'clustering_mode': approach.__class__.__name__,
                'clustering_technique': clustering_technique.__class__.__name__,
                'dimensionality_reduction_techniques': dimensionality_reduction_techniques,
                'score': round(score, 3),
                'contributions': contributions,
                'clusters': Clustering().from_membership_list(clusters).to_cluster_list(),
                'projections': projections,
                'time_contributions': round(contributions_time, 5),
                'time_clustering': round(cluster_time, 5)
            })

            if verbose:
                show_progress(iteration, iterations)
        print()

    df = pd.DataFrame(data)
    return df
