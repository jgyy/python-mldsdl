"""
T-Tests and P-Values
"""
from numpy.random import normal
from scipy.stats import ttest_ind


def wrapper():
    """
    wrapper function
    """
    adata = normal(25.0, 5.0, 10000)
    bdata = normal(26.0, 5.0, 10000)
    print(ttest_ind(adata, bdata))
    bdata = normal(25.0, 5.0, 10000)
    print(ttest_ind(adata, bdata))
    adata = normal(25.0, 5.0, 100000)
    bdata = normal(25.0, 5.0, 100000)
    print(ttest_ind(adata, bdata))
    adata = normal(25.0, 5.0, 1000000)
    bdata = normal(25.0, 5.0, 1000000)
    print(ttest_ind(adata, bdata))
    print(ttest_ind(adata, adata))


if __name__ == "__main__":
    wrapper()
