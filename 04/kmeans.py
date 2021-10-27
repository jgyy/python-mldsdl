"""
K-Means Clustering Example
"""
from numpy import array
from numpy.random import seed, uniform, normal
from matplotlib.pyplot import figure, scatter, show
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale


def create_clustered_data(num, kmean):
    """
    Create fake income/age clusters for N people in k clusters
    """
    seed(10)
    points_per_cluster = float(num) / kmean
    xdata = []
    for _ in range(kmean):
        income_centroid = uniform(20000.0, 200000.0)
        age_centroid = uniform(20.0, 70.0)
        for _ in range(int(points_per_cluster)):
            xdata.append([normal(income_centroid, 10000.0), normal(age_centroid, 2.0)])
    xdata = array(xdata)
    return xdata


def wrapper():
    """
    Note I'm scaling the data to normalize it! Important for good results.
    We can look at the clusters each data point was assigned to
    And we'll visualize it:
    """
    data = create_clustered_data(100, 5)
    model = KMeans(n_clusters=5)
    model = model.fit(scale(data))
    print(model.labels_)
    figure(figsize=(8, 6))
    scatter(data[:, 0], data[:, 1], c=model.labels_.astype(float))


if __name__ == "__main__":
    wrapper()
    show()
