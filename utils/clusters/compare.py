import time
from typing import Callable

import numpy as np
import pandas as pd
from clusim.clustering import Clustering
from tqdm import tqdm, trange

from config.config_dirs import HEATMAPS_DATA_RAW
from config.config_heatmaps import CLUSTERING_TECHNIQUE
from utils.clusters.Approach import Approach


def compare_approaches(
        approaches: list,
        iterations: int,
        get_info: Callable[[Approach], str] = None
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
    df = pd.DataFrame()
    for idx, approach in (bar := tqdm(list(enumerate(approaches)))):
        bar.set_description(f'Comparing the approaches ({approach})')
        # Extract some information about the current approach
        explainer = approach.get_explainer()
        dimensionality_reduction_techniques = approach.get_dimensionality_reduction_techniques()
        for _ in trange(iterations, desc='Iterating', leave=False):
            # Generate the contributions
            contributions_start = time.time()
            contributions = approach.generate_contributions()
            contributions_time = time.time() - contributions_start
            if len(contributions) == 0:
                # Impossible to generate the contributions -> skip
                continue
            # Cluster the contributions
            cluster_start = time.time()
            try:
                clusters, projections, score = approach.cluster_contributions(contributions)
            except ValueError:
                # No clusters -> silhouette error
                clusters, score, projections = [], np.nan, []
            cluster_time = time.time() - cluster_start

            if len(clusters) == 0:
                # No cluster found -> skip
                continue

            # Collect the information for the run
            data.append({
                'approach': explainer.__class__.__name__,
                'clustering_mode': approach.__class__.__name__,
                'clustering_technique': CLUSTERING_TECHNIQUE.__qualname__,
                'dimensionality_reduction_techniques': dimensionality_reduction_techniques,
                'score': round(score, 3),
                'contributions': contributions,
                'clusters': Clustering().from_membership_list(clusters).to_cluster_list(),
                'projections': projections,
                'time_contributions': round(contributions_time, 5),
                'time_clustering': round(cluster_time, 5)
            })
        df = pd.DataFrame(data)
        df.to_pickle(HEATMAPS_DATA_RAW)
    return df
