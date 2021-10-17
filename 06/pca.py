"""
Principal Component Analysis
"""
from types import SimpleNamespace
from itertools import cycle
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from matplotlib.pyplot import figure, show, scatter, legend


def wrapper():
    """
    wrapper function
    """
    iris = SimpleNamespace(**load_iris())
    num_samples, num_features = iris.data.shape
    print(num_samples)
    print(num_features)
    print(list(iris.target_names))
    xdata = iris.data
    pca = PCA(n_components=2, whiten=True).fit(xdata)
    x_pca = pca.transform(xdata)
    print(pca.components_)
    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))
    colors = cycle("rgb")
    target_ids = range(len(iris.target_names))
    figure()
    for i, color, label in zip(target_ids, colors, iris.target_names):
        scatter(
            x_pca[iris.target == i, 0], x_pca[iris.target == i, 1], c=color, label=label
        )
    legend()


if __name__ == "__main__":
    wrapper()
    show()
