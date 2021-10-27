"""
Seaborn
"""
from warnings import filterwarnings
from seaborn import (
    histplot,
    pairplot,
    scatterplot,
    jointplot,
    lmplot,
    boxplot,
    swarmplot,
    countplot,
    heatmap,
    set as snsset
)
from pandas import DataFrame, read_csv
from matplotlib.pyplot import figure, show


def wrapper():
    """
    wrapper function
    """
    filterwarnings("ignore", category=UserWarning)
    dframe = DataFrame(
        read_csv("http://media.sundog-soft.com/SelfDriving/FuelEfficiency.csv")
    )
    figure()
    gear_counts = dframe["# Gears"].value_counts()
    gear_counts.plot(kind="bar")
    figure()
    snsset()
    gear_counts.plot(kind="bar")
    print(dframe.head())
    figure()
    histplot(dframe["CombMPG"], kde=True)
    df2 = dframe[["Cylinders", "CityMPG", "HwyMPG", "CombMPG"]]
    print(df2.head())
    pairplot(df2, height=2.5)
    figure()
    scatterplot(x="Eng Displ", y="CombMPG", data=dframe)
    jointplot(x="Eng Displ", y="CombMPG", data=dframe)
    lmplot(x="Eng Displ", y="CombMPG", data=dframe)
    figure()
    snsset(rc={"figure.figsize": (15, 5)})
    axe = boxplot(x="Mfr Name", y="CombMPG", data=dframe)
    axe.set_xticklabels(axe.get_xticklabels(), rotation=45)
    figure()
    axe = swarmplot(x="Mfr Name", y="CombMPG", data=dframe)
    axe.set_xticklabels(axe.get_xticklabels(), rotation=45)
    figure()
    axe = countplot(x="Mfr Name", data=dframe)
    axe.set_xticklabels(axe.get_xticklabels(), rotation=45)
    figure()
    df2 = dframe.pivot_table(
        index="Cylinders", columns="Eng Displ", values="CombMPG", aggfunc="mean"
    )
    heatmap(df2)
    figure()
    scatterplot(x="# Gears", y="CombMPG", data=dframe)
    lmplot(x="# Gears", y="CombMPG", data=dframe)
    jointplot(x="# Gears", y="CombMPG", data=dframe)
    figure()
    boxplot(x="# Gears", y="CombMPG", data=dframe)
    figure()
    swarmplot(x="# Gears", y="CombMPG", data=dframe)


if __name__ == "__main__":
    wrapper()
    show()
