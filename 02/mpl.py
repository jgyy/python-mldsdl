"""
MatPlotLib Basics
"""
from os.path import join, dirname
from numpy import arange, concatenate, ones
from numpy.random import normal, rand
from scipy.stats import norm
from pylab import randn
from matplotlib.pyplot import (
    figure,
    show,
    plot,
    savefig,
    axes,
    title,
    xlabel,
    ylabel,
    legend,
    xkcd,
    xticks,
    yticks,
    annotate,
    rcdefaults,
    pie,
    bar,
    hist,
    boxplot,
    scatter,
)


def line_graph():
    """
    Draw a line graph
    """
    xdata = arange(-3, 3, 0.01)
    figure()
    plot(xdata, norm.pdf(xdata))

    figure()
    plot(xdata, norm.pdf(xdata))
    plot(xdata, norm.pdf(xdata, 1.0, 0.5))

    figure()
    plot(xdata, norm.pdf(xdata))
    plot(xdata, norm.pdf(xdata, 1.0, 0.5))
    savefig(join(dirname(__file__), "MyPlot.png"), format="png")

    figure()
    axe = axes()
    axe.set_xlim([-5, 5])
    axe.set_ylim([0, 1.0])
    axe.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    axe.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plot(xdata, norm.pdf(xdata))
    plot(xdata, norm.pdf(xdata, 1.0, 0.5))

    figure()
    axe = axes()
    axe.set_xlim([-5, 5])
    axe.set_ylim([0, 1.0])
    axe.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    axe.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axe.grid()
    plot(xdata, norm.pdf(xdata))
    plot(xdata, norm.pdf(xdata, 1.0, 0.5))

    figure()
    axe = axes()
    axe.set_xlim([-5, 5])
    axe.set_ylim([0, 1.0])
    axe.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    axe.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axe.grid()
    plot(xdata, norm.pdf(xdata), "b-")
    plot(xdata, norm.pdf(xdata, 1.0, 0.5), "r:")


def label_axes():
    """
    Labeling Axes and Adding a Legend
    """
    xdata = arange(-3, 3, 0.01)
    figure()
    axe = axes()
    axe.set_xlim([-5, 5])
    axe.set_ylim([0, 1.0])
    axe.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    axe.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axe.grid()
    xlabel("Greebles")
    ylabel("Probability")
    plot(xdata, norm.pdf(xdata), "b-")
    plot(xdata, norm.pdf(xdata, 1.0, 0.5), "r:")
    legend(["Sneetches", "Gacks"], loc=4)

    fig = figure()
    xkcd()
    axe = fig.add_subplot(1, 1, 1)
    axe.spines["right"].set_color("none")
    axe.spines["top"].set_color("none")
    xticks([])
    yticks([])
    axe.set_ylim([-30, 10])
    data = ones(100)
    data[70:] -= arange(30)
    annotate(
        "THE DAY I REALIZED\nI COULD COOK BACON\nWHENEVER I WANTED",
        xy=(70, 1),
        arrowprops=dict(arrowstyle="->"),
        xytext=(15, -10),
    )
    plot(data)
    xlabel("time")
    ylabel("my overall health")

    figure()
    rcdefaults()
    values = [12, 55, 4, 32, 14]
    colors = ["r", "g", "b", "c", "m"]
    explode = [0, 0, 0.2, 0, 0]
    labels = ["India", "United States", "Russia", "China", "Europe"]
    pie(values, colors=colors, labels=labels, explode=explode)
    title("Student Locations")
    figure()
    values = [12, 55, 4, 32, 14]
    colors = ["r", "g", "b", "c", "m"]
    bar(range(0, 5), values, color=colors)

    figure()
    xdata = randn(500)
    ydata = randn(500)
    scatter(xdata, ydata)

    figure()
    incomes = normal(27000, 15000, 10000)
    hist(incomes, 50)

    figure()
    uniform_skewed = rand(100) * 100 - 40
    high_outliers = rand(10) * 50 + 100
    low_outliers = rand(10) * -50 - 100
    data = concatenate((uniform_skewed, high_outliers, low_outliers))
    boxplot(data)


if __name__ == "__main__":
    line_graph()
    label_axes()
    show()
