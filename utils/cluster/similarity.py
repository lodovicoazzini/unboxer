from clusim.clustering import Clustering
from clusim.sim import nmi


def cluster_similarity(lhs: list, rhs: list, metric: callable = nmi):
    c1 = Clustering().from_membership_list(lhs)
    c2 = Clustering().from_membership_list(rhs)
    return metric(c1, c2)
