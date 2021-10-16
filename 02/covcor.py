"""
Covariance and Correlation
"""
from numpy import corrcoef
from numpy.random import normal
from pylab import mean, dot, scatter, figure, show


def wrapper():
    """
    wrapper function
    """
    de_mean = lambda xdata: [xi - mean(xdata) for xi in xdata]
    covariance = lambda xdata, ydata: dot(de_mean(xdata), de_mean(ydata)) / (
        len(xdata) - 1
    )
    figure()
    page_speeds = normal(3.0, 1.0, 1000)
    purchase_amount = normal(50.0, 10.0, 1000)
    scatter(page_speeds, purchase_amount)
    print(covariance(page_speeds, purchase_amount))
    figure()
    purchase_amount = normal(50.0, 10.0, 1000) / page_speeds
    scatter(page_speeds, purchase_amount)
    print(covariance(page_speeds, purchase_amount))
    correlation = (
        lambda xdata, ydata: covariance(xdata, ydata) / xdata.std() / ydata.std()
    )
    print(correlation(page_speeds, purchase_amount))
    print(corrcoef(page_speeds, purchase_amount))
    figure()
    purchase_amount = 100 - page_speeds * 3
    scatter(page_speeds, purchase_amount)
    print(correlation(page_speeds, purchase_amount))


if __name__ == "__main__":
    wrapper()
    show()
