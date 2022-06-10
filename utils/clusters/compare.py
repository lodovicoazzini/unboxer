import time
from typing import Callable

import numpy as np
import pandas as pd
from clusim.clustering import Clustering

from utils.clusters.ClusteringMode import ClusteringMode
from utils.general import show_progress


def compare_approaches(
        approaches: list,
        iterations: int,
        get_info: Callable[[ClusteringMode], str] = None
) -> pd.DataFrame:
    """
    Compare a list of approaches and return the collected data
    :param approaches: The list of approaches to compare
    :param iterations: The number of iterations to process each approach
    :param get_info: A function to get the string information about the current approach
    :return: A dataframe with the data collected during the tries
    """
    # Iterate over the approaches in the list
    data = []
    for idx, approach in enumerate(approaches):
        # Extract some information about the current approach
        explainer = approach.get_explainer()
        clustering_technique = approach.get_clustering_technique()
        dimensionality_reduction_techniques = approach.get_dimensionality_reduction_techniques()

        # Generate the contributions
        contributions_start = time.time()
        contributions = approach.generate_contributions()
        contributions_time = time.time() - contributions_start

        def execution(_):
            # Cluster the contributions
            cluster_start = time.time()
            try:
                clusters, projections, score = approach.cluster_contributions(contributions)
            except ValueError:
                # No clusters -> silhouette error
                clusters, score, projections = [], np.nan, []
            cluster_time = time.time() - cluster_start

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

        message = f'{explainer.__class__.__name__} ({get_info(approach)})'

        show_progress(execution=execution, iterable=range(iterations), message=message)

    df = pd.DataFrame(data)
    return df
