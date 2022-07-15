import numpy as np
from tqdm import tqdm


def compute_map(features, samples):
    # Log the information
    feature_comb_str = "+".join([feature.feature_name for feature in features])
    print(f'Using the features {feature_comb_str}')

    # Compute the shape of the map (number of cells of each feature)
    shape = [feature.num_cells for feature in features]
    # Keep track of the samples in each cell, initialize a matrix of empty arrays
    archive_data = np.empty(shape=shape, dtype=list)
    for idx in np.ndindex(*archive_data.shape):
        archive_data[idx] = []
    # Count the number of items in each cell
    coverage_data = np.zeros(shape=shape, dtype=int)
    misbehaviour_data = np.zeros(shape=shape, dtype=int)
    # Initialize the matrix of clusters to empty lists
    clusters = np.empty(shape=shape, dtype=list)
    for idx in np.ndindex(*clusters.shape):
        clusters[idx] = []

    for idx, sample in enumerate(samples):
        # Coordinates reason in terms of bins 1, 2, 3, while data is 0-indexed
        coords = tuple([feature.get_coordinate_for(sample) - 1 for feature in features])
        # Archive the sample

        archive_data[coords].append(sample)
        # Increment the coverage
        coverage_data[coords] += 1
        # Increment the misbehaviour
        if sample.is_misbehavior:
            misbehaviour_data[coords] += 1
        # Update the clusters
        clusters[coords].append(idx)

    return archive_data, coverage_data, misbehaviour_data, clusters
