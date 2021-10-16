"""
Linear Regression
"""
from numpy.random import normal
from matplotlib.pyplot import figure, show, scatter, plot
from scipy.stats import linregress


def wrapper():
    """
    wrapper function
    """
    page_speeds = normal(3.0, 1.0, 1000)
    purchase_amount = 100 - (page_speeds + normal(0, 0.1, 1000)) * 3
    figure()
    scatter(page_speeds, purchase_amount)
    slope, intercept, r_value, _, _ = linregress(page_speeds, purchase_amount)
    print(r_value ** 2)

    figure()
    predict = lambda x: slope * x + intercept
    fit_line = predict(page_speeds)
    scatter(page_speeds, purchase_amount)
    plot(page_speeds, fit_line, c="r")


if __name__ == "__main__":
    wrapper()
    show()
