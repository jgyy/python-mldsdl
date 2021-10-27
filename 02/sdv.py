"""
Standard Deviation and Variance
"""
from numpy.random import normal
from matplotlib.pyplot import hist, figure, show


def wrapper():
    """
    wrapper function
    """
    incomes = normal(100.0, 50.0, 10000)
    figure()
    hist(incomes, 50)
    print(incomes.std())
    print(incomes.var())


if __name__ == "__main__":
    wrapper()
    show()
