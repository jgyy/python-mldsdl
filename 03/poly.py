"""
Polynomial Regression
"""
from matplotlib.pyplot import figure, show, scatter, plot
from numpy import array, poly1d, polyfit, linspace
from numpy.random import seed, normal
from sklearn.metrics import r2_score


def wrapper():
    """
    wrapper function
    """
    seed(2)
    page_speeds = normal(3.0, 1.0, 1000)
    purchase_amount = normal(50.0, 10.0, 1000) / page_speeds
    figure()
    scatter(page_speeds, purchase_amount)
    xdata = array(page_speeds)
    ydata = array(purchase_amount)
    poly4 = poly1d(polyfit(xdata, ydata, 4))
    figure()
    xpoly = linspace(0, 7, 100)
    scatter(xdata, ydata)
    plot(xpoly, poly4(xpoly), c="r")
    root2 = r2_score(ydata, poly4(xdata))
    print(root2)


if __name__ == "__main__":
    wrapper()
    show()
