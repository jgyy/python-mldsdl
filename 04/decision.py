"""
Decision Trees
"""
from io import StringIO, BytesIO
from os.path import dirname, join
from pandas import DataFrame, read_csv
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from pydotplus import graph_from_dot_data
from PIL import Image


def wrapper():
    """
    wrapper function
    """
    input_file = join(dirname(__file__), "PastHires.csv")
    dframe = DataFrame(read_csv(input_file, header=0))
    print(dframe.head())
    data = {"Y": 1, "N": 0}
    dframe["Hired"] = dframe["Hired"].map(data)
    dframe["Employed?"] = dframe["Employed?"].map(data)
    dframe["Top-tier school"] = dframe["Top-tier school"].map(data)
    dframe["Interned"] = dframe["Interned"].map(data)
    data = {"BS": 0, "MS": 1, "PhD": 2}
    dframe["Level of Education"] = dframe["Level of Education"].map(data)
    print(dframe.head())
    features = list(dframe.columns[:6])
    print(features)
    ydata = dframe["Hired"]
    xdata = dframe[features]
    clf = DecisionTreeClassifier()
    clf = clf.fit(xdata, ydata)
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data, feature_names=features)
    graph = graph_from_dot_data(dot_data.getvalue())
    img = Image.open(BytesIO(graph.create_png()))
    img.show()
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(xdata.values, ydata.values)
    print(clf.predict([[10, 1, 4, 0, 0, 0]]))
    print(clf.predict([[10, 0, 4, 0, 0, 0]]))


if __name__ == "__main__":
    wrapper()
