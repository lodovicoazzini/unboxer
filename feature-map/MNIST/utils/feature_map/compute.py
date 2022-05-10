import numpy as np

from feature import Feature


def compute_map_3d(features, samples):
    # Generate the map axes
    map_features = []
    for f in features:
        print("Using feature %s" % f[0])
        map_features.append(Feature(f[0], f[1], f[2], f[3], f[4]))

    feature1 = map_features[0]
    feature2 = map_features[1]
    feature3 = map_features[2]

    # Reshape the data as ndimensional array. But account for the lower and upper bins.
    archive_data = np.full([feature1.num_cells, feature2.num_cells, feature3.num_cells], None, dtype=object)
    # counts the number of samples in each cell
    coverage_data = np.zeros(shape=(10, feature1.num_cells, feature2.num_cells, feature3.num_cells), dtype=int)

    misbehaviour_data = np.zeros(shape=(feature1.num_cells, feature2.num_cells, feature3.num_cells), dtype=int)

    for sample in samples:
        # Coordinates reason in terms of bins 1, 2, 3, while data is 0-indexed
        x_coord = feature1.get_coordinate_for(sample) - 1
        y_coord = feature2.get_coordinate_for(sample) - 1
        z_coord = feature3.get_coordinate_for(sample) - 1

        if archive_data[x_coord, y_coord, z_coord] is None:
            arx = [sample]
            archive_data[x_coord, y_coord, z_coord] = arx
        else:
            archive_data[x_coord, y_coord, z_coord].append(sample)
        # Increment the coverage
        coverage_data[int(sample.expected_label), x_coord, y_coord, z_coord] += 1

        if sample.is_misbehavior:
            # Increment the misbehaviour
            misbehaviour_data[x_coord, y_coord, z_coord] += 1

    return archive_data, coverage_data, misbehaviour_data


def compute_map(features, samples):
    feature_comb_str = "+".join([feature.feature_name for feature in features])
    print(f'Using the features {feature_comb_str}')
    feature1 = features[0]
    feature2 = features[1]

    # Keep track of the samples in each cell
    archive_data = np.empty([feature1.num_cells, feature2.num_cells], dtype=list)
    # Count the number of items in each cell
    coverage_data = np.zeros(shape=(feature1.num_cells, feature2.num_cells), dtype=int)
    misbehaviour_data = np.zeros(shape=(feature1.num_cells, feature2.num_cells), dtype=int)
    # Keep track of the seeds clusters
    clusters = np.empty([feature1.num_cells, feature2.num_cells], dtype=list)

    for idx, sample in enumerate(samples):
        x_coord = feature1.get_coordinate_for(sample) - 1
        y_coord = feature2.get_coordinate_for(sample) - 1

        # Archive the sample
        if type(archive_data[x_coord, y_coord]) is list:
            archive_data[x_coord, y_coord].append(sample)
        else:
            archive_data[x_coord, y_coord] = [sample]
        # Increment the coverage
        coverage_data[x_coord, y_coord] += 1
        # Increment the misbehaviour
        if sample.is_misbehavior:
            misbehaviour_data[x_coord, y_coord] += 1
        # Update teh clusters
        if type(clusters[x_coord, y_coord]) is list:
            clusters[x_coord, y_coord].append(idx)
        else:
            clusters[x_coord, y_coord] = [idx]

    return archive_data, coverage_data, misbehaviour_data, clusters
