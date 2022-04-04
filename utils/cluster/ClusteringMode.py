import abc

from sklearn.metrics import silhouette_score

from utils.cluster.preprocessing import generate_contributions


class ClusteringMode(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, explainer, dim_red_techs, clus_tech, mask):
        self.explainer = explainer
        self.dim_red_techs = dim_red_techs
        self.clus_tech = clus_tech
        self.mask = mask

    @abc.abstractmethod
    def generate_contributions(self, test_data, predictions):
        raise NotImplementedError()

    @abc.abstractmethod
    def cluster_contributions(self, contributions) -> tuple[list, float, list]:
        raise NotImplementedError()

    def get_clustering_technique(self):
        return self.clus_tech

    def get_dimensionality_reduction_techniques(self):
        return self.dim_red_techs

    def get_explainer(self):
        return self.explainer


class LocalLatentMode(ClusteringMode):

    def __init__(self, explainer, dim_red_techs, clus_tech, mask):
        super(LocalLatentMode, self).__init__(explainer, dim_red_techs, clus_tech, mask)

    def generate_contributions(self, test_data, predictions):
        # generate the contributions only for the filtered ones
        return generate_contributions(self.explainer, test_data[self.mask], predictions[self.mask])

    def cluster_contributions(self, contributions) -> tuple[list, float, list]:
        # projecting the contributions in the latent space
        # initial transformation to flatten the contributions
        projections = contributions.reshape(contributions.shape[0], -1)
        for dim_red_tech in self.dim_red_techs:
            projections = dim_red_tech.fit_transform(projections)

        clusters = self.clus_tech.fit_predict(projections)
        # compute the silhouette score for the clusters
        score = silhouette_score(projections, clusters)

        return clusters, score, projections


class GlobalLatentMode(ClusteringMode):

    def __init__(self, explainer, dim_red_techs, clus_tech, mask):
        super(GlobalLatentMode, self).__init__(explainer, dim_red_techs, clus_tech, mask)

    def generate_contributions(self, test_data, predictions):
        return generate_contributions(self.explainer, test_data, predictions)

    def cluster_contributions(self, contributions) -> tuple[list, float, list]:
        # projecting the contributions in the latent space
        # initial transformation to flatten the contributions
        projections = contributions.reshape(contributions.shape[0], -1)
        for dim_red_tech in self.dim_red_techs:
            projections = dim_red_tech.fit_transform(projections)

        # cluster the filtered contributions
        clusters = self.clus_tech.fit_predict(projections[self.mask])
        # compute the silhouette score for the clusters
        score = silhouette_score(projections[self.mask], clusters)

        return clusters, score, projections[self.mask]
