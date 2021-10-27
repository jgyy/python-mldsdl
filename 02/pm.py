"""
Percentiles Moments: Mean, Variance, Skew, Kurtosis
"""
from numpy import percentile, mean, var
from numpy.random import normal
from matplotlib.pyplot import figure, show, hist
from scipy import stats


def wrapper():
    """
    wrapper function
    """
    vals = normal(0, 0.5, 10000)
    figure()
    hist(vals, 50)
    print(percentile(vals, 50))
    print(percentile(vals, 90))
    print(percentile(vals, 20))
    vals = normal(0, 0.5, 10000)
    figure()
    hist(vals, 50)
    print(mean(vals))
    print(var(vals))
    print(stats.skew(vals))
    print(stats.kurtosis(vals))


if __name__ == "__main__":
    wrapper()
    show()
