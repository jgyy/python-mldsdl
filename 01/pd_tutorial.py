"""
Introducing Pandas
"""
from os.path import join, dirname
from pandas import DataFrame, read_csv
from matplotlib.pyplot import figure, show


def wrapper():
    """
    wrapper function
    """
    past_hires = join(dirname(__file__), "PastHires.csv")
    dframe = DataFrame(read_csv(past_hires))
    print(dframe.head())
    print(dframe.head(10))
    print(dframe.tail(4))
    print(dframe.shape)
    print(dframe.size)
    print(len(dframe))
    print(dframe.columns)
    print(dframe["Hired"])
    print(dframe["Hired"][:5])
    print(dframe["Hired"][5])
    print(dframe[["Years Experience", "Hired"]])
    print(dframe[["Years Experience", "Hired"]][:5])
    print(dframe.sort_values(["Years Experience"]))
    degree_counts = dframe["Level of Education"].value_counts()
    print(degree_counts)
    figure()
    degree_counts.plot(kind="bar")


if __name__ == "__main__":
    wrapper()
    show()
