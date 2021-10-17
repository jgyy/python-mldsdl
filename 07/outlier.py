"""
Dealing with Outliers
"""
from numpy import append, median, std, mean
from numpy.random import normal
from matplotlib.pyplot import figure, show, hist


def reject_outliers(data):
    """
    reject outliers function
    """
    udata = median(data)
    sdata = std(data)
    filtered = [e for e in data if udata - 2 * sdata < e < udata + 2 * sdata]
    return filtered


def wrapper():
    """
    wrapper function
    """
    incomes = normal(27000, 15000, 10000)
    incomes = append(incomes, [1000000000])

    figure()
    hist(incomes, 50)
    print(incomes.mean())

    figure()
    filtered = reject_outliers(incomes)
    hist(filtered, 50)
    print(mean(filtered))


if __name__ == "__main__":
    wrapper()
    show()
