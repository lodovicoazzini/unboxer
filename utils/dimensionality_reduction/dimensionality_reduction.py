import numpy as np
from sklearn.manifold import TSNE


def tsne(heatmaps: np.ndarray) -> np.ndarray:
    return TSNE(n_components=2).fit_transform(heatmaps)
