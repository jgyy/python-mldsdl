"""
Train / Test
"""
from numpy import array, poly1d, polyfit, linspace
from numpy.random import seed, normal
from matplotlib.pyplot import figure, show, scatter, axes, plot
from sklearn.metrics import r2_score


def wrapper():
    """
    wrapper function
    """
    seed(2)
    page_speeds = normal(3.0, 1.0, 100)
    purchase_amount = normal(50.0, 30.0, 100) / page_speeds

    figure()
    scatter(page_speeds, purchase_amount)
    trainx = page_speeds[:80]
    testx = page_speeds[80:]
    trainy = purchase_amount[:80]
    testy = purchase_amount[80:]

    figure()
    scatter(trainx, trainy)

    figure()
    scatter(testx, testy)
    xdata = array(trainx)
    ydata = array(trainy)
    poly4 = poly1d(polyfit(xdata, ydata, 8))

    figure()
    xpoly = linspace(0, 7, 100)
    axe = axes()
    axe.set_xlim([0, 7])
    axe.set_ylim([0, 200])
    scatter(xdata, ydata)
    plot(xpoly, poly4(xpoly), c="r")

    figure()
    testx = array(testx)
    testy = array(testy)
    axe = axes()
    axe.set_xlim([0, 7])
    axe.set_ylim([0, 200])
    scatter(testx, testy)
    plot(xpoly, poly4(xpoly), c="r")
    r_2 = r2_score(testy, poly4(testx))
    print(r_2)
    r_2 = r2_score(array(trainy), poly4(array(trainx)))
    print(r_2)


if __name__ == "__main__":
    wrapper()
    show()
