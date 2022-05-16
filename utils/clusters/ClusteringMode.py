import abc

import numpy as np
import tensorflow as tf
from sklearn.metrics import silhouette_score

from utils import global_values


class ClusteringMode(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, explainer, dimensionality_reduction_techniques, clustering_technique):
        self.__explainer = explainer
        self.__dimensionality_reduction_techniques = dimensionality_reduction_techniques
        self.__clustering_technique = clustering_technique

    @abc.abstractmethod
    def generate_contributions(self) -> np.ndarray:
        """
        Generate the contributions for the predictions
        :return: The contributions for the predictions
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def cluster_contributions(self, contributions: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Cluster the contributions
        :param contributions: The contributions
        :return: The clusters for the contributions as membership list
        """
        raise NotImplementedError()

    def get_clustering_technique(self):
        return self.__clustering_technique

    def get_dimensionality_reduction_techniques(self):
        return self.__dimensionality_reduction_techniques

    def get_explainer(self):
        return self.__explainer

    def _generate_contributions(
            self,
            data: np.ndarray,
            predictions: np.ndarray,
            only_positive: bool = True
    ) -> np.ndarray:
        # Generate the contributions
        contributions = self.__explainer.explain(data, predictions)
        # Convert the contributions to grayscale
        try:
            contributions = np.squeeze(tf.image.rgb_to_grayscale(contributions).numpy())
        except tf.errors.InvalidArgumentError:
            pass
        # Filter for the positive contributions
        if only_positive:
            contributions = np.ma.masked_less(np.squeeze(contributions), 0).filled(0)
        return contributions


class LocalLatentMode(ClusteringMode):

    def __init__(self, explainer, dimensionality_reduction_techniques, clustering_technique):
        super(LocalLatentMode, self).__init__(explainer, dimensionality_reduction_techniques, clustering_technique)

    def generate_contributions(self):
        # Generate the contributions for the filtered data
        return super(LocalLatentMode, self)._generate_contributions(
            global_values.test_data[global_values.mask_label],
            global_values.predictions_cat[global_values.mask_label]
        )

    def cluster_contributions(self, contributions: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        # Flatten teh contributions and project then in the latent space
        contributions_flattened = contributions.reshape(contributions.shape[0], -1)
        projections = np.array([])
        for dim_red_tech in self.get_dimensionality_reduction_techniques():
            projections = dim_red_tech.fit_transform(contributions_flattened)
        # Cluster the projections
        clusters = self.get_clustering_technique().fit_predict(projections)
        # Compute the silhouette for the clusters
        try:
            score = silhouette_score(projections, clusters)
        except ValueError:
            score = np.nan
        return clusters, projections, score


class GlobalLatentMode(ClusteringMode):

    def __init__(self, explainer, dimensionality_reduction_techniques, clustering_technique):
        super(GlobalLatentMode, self).__init__(explainer, dimensionality_reduction_techniques, clustering_technique, )

    def generate_contributions(self):
        # Generate the contributions for the whole data
        return super(GlobalLatentMode, self)._generate_contributions(
            global_values.test_data,
            global_values.predictions
        )

    def cluster_contributions(self, contributions: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        # Flatten the contributions and project them into the latent space
        contributions_flattened = contributions.reshape(contributions.shape[0], -1)
        projections = np.array([])
        for dim_red_tech in self.get_dimensionality_reduction_techniques():
            projections = dim_red_tech.fit_transform(contributions_flattened)
        # Cluster the filtered projections
        projections_filtered = projections[global_values.mask_label]
        clusters = self.get_clustering_technique().fit_predict(projections_filtered)
        # Compute the silhouette score for the clusters
        try:
            score = silhouette_score(projections_filtered, clusters)
        except ValueError:
            score = np.nan
        return clusters, projections_filtered, score
