"""
Mean, Median, Mode, and introducing NumPy
"""
from numpy import mean, median, append
from numpy.random import normal, randint
from matplotlib.pyplot import hist, figure, show
from scipy.stats import mode


def wrapper():
    """
    wrapper function
    """
    incomes = normal(27000, 15000, 10000)
    print(mean(incomes))
    figure()
    hist(incomes, 50)
    print(median(incomes))
    incomes = append(incomes, [1000000000])
    print(median(incomes))
    print(mean(incomes))
    ages = randint(18, high=90, size=500)
    print(ages)
    print(mode(ages))


if __name__ == "__main__":
    wrapper()
    show()
