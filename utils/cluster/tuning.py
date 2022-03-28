import numpy as np
from sklearn.metrics import silhouette_score

from utils.cluster.preprocessing import distance_matrix
from utils.image_similarity.intensity_based import euclidean_distance


def find_optimal_configuration(
        models, data,
        distance_metric='euclidean',
        dist_func=euclidean_distance,
        verbose=False
):
    # if the distance metric is precomputed -> compute the distance matrix
    if distance_metric == 'precomputed':
        dist_data = distance_matrix(data, dist_func=dist_func)
    else:
        dist_data = data
    # compute the silhouette scores for all the configurations
    scores = np.array([])
    for model in models:
        clusters = model.fit_predict(data)
        score = silhouette_score(dist_data, clusters, metric=distance_metric)
        scores = np.append(scores, score)

    # find the optimal configuration
    optimal = models[np.argmax(scores)]

    if verbose:
        print(f"""
Found optimal configuration with score: {np.amax(scores)}
and parameters {optimal.get_params()}
        """)

    # return the optimal configuration
    return optimal
