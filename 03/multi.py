"""
Multiple Regression
"""
from warnings import filterwarnings
from pandas import DataFrame, read_excel, cut
from matplotlib.pyplot import figure, show
from numpy import arange, insert
from statsmodels.api import add_constant, OLS
from sklearn.preprocessing import StandardScaler


def wrapper():
    """
    wrapper function
    """
    filterwarnings("ignore", category=FutureWarning)
    dframe = DataFrame(
        read_excel("http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls")
    )
    df1 = dframe.loc[:, ["Mileage", "Price"]]
    bins = arange(0, 50000, 10000)
    groups = df1.groupby(cut(df1["Mileage"], bins)).mean()
    print(groups.head())
    figure()
    groups["Price"].plot.line()
    scale = StandardScaler()
    xdata = dframe.loc[:, ["Mileage", "Cylinder", "Doors"]]
    ydata = dframe["Price"]
    xdata[["Mileage", "Cylinder", "Doors"]] = scale.fit_transform(
        xdata[["Mileage", "Cylinder", "Doors"]].values
    )
    xdata = add_constant(xdata)
    print(xdata)
    est = OLS(ydata, xdata).fit()
    print(est.summary())
    print(ydata.groupby(dframe.Doors).mean())
    scaled = scale.transform([[45000, 8, 4]])
    scaled = insert(scaled[0], 0, 1)
    print(scaled)
    predicted = est.predict(scaled)
    print(predicted)


if __name__ == "__main__":
    wrapper()
    show()
