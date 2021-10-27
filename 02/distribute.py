"""
Examples of Data Distributions
"""
from numpy import arange
from numpy.random import uniform, normal
from matplotlib.pyplot import figure, show, hist, plot
from scipy.stats import norm, expon, binom, poisson


def wrapper():
    """
    wrapper function
    """
    values = uniform(-10.0, 10.0, 100000)
    figure()
    hist(values, 50)
    xdata = arange(-3, 3, 0.001)
    figure()
    plot(xdata, norm.pdf(xdata))
    muu = 5.0
    sigma = 2.0
    values = normal(muu, sigma, 10000)
    figure()
    hist(values, 50)
    xdata = arange(0, 10, 0.001)
    figure()
    plot(xdata, expon.pdf(xdata))
    num, pro = 10, 0.5
    xdata = arange(0, 10, 0.001)
    figure()
    plot(xdata, binom.pmf(xdata, num, pro))
    muu = 500
    xdata = arange(400, 600, 0.5)
    figure()
    plot(xdata, poisson.pmf(xdata, muu))


if __name__ == "__main__":
    wrapper()
    show()
