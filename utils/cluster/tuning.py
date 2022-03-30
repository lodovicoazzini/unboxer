import numpy as np
from sklearn.metrics import silhouette_score


def find_optimal_configuration(
        models, data,
        verbose=False,
        silhouette_metric='euclidean'
):
    # compute the silhouette scores for all the configurations
    scores = np.array([])
    for model in models:
        clusters = model.fit_predict(data)
        try:
            score = silhouette_score(data, clusters, metric=silhouette_metric)
            scores = np.append(scores, score)
        except ValueError:
            # no cluster identified
            continue

    # find the optimal configuration
    optimal = models[np.argmax(scores)]

    if verbose:
        print(f"""
Found optimal configuration with score: {np.amax(scores)}
and parameters {optimal.get_params()}
        """)

    # return the optimal configuration
    return optimal
