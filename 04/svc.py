"""
Support Vector Machines
"""
from numpy import array, float as npfloat, meshgrid, arange, c_
from numpy.random import seed, uniform, normal
from matplotlib.pyplot import figure, show, scatter, contourf, cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


def create_clustered_data(num, kmean):
    """
    Create fake income/age clusters for N people in k clusters
    """
    seed(1234)
    points_per_cluster = float(num) / kmean
    xdata = []
    ydata = []
    for i in range(kmean):
        income_centroid = uniform(20000.0, 200000.0)
        age_centroid = uniform(20.0, 70.0)
        for _ in range(int(points_per_cluster)):
            xdata.append([normal(income_centroid, 10000.0), normal(age_centroid, 2.0)])
            ydata.append(i)
    xdata = array(xdata)
    ydata = array(ydata)
    return xdata, ydata


def plot_predictions(clf, xdata, ydata):
    """
    Create a dense grid of points to sample
    Convert to Numpy arrays
    Convert to a list of 2D (income, age) points
    Generate predicted labels (cluster numbers) for each point
    Reshape results to match xx dimension
    Draw the contour > Draw the points
    """
    xxx, yyy = meshgrid(arange(-1, 1, 0.001), arange(-1, 1, 0.001))
    npx = xxx.ravel()
    npy = yyy.ravel()
    sample_points = c_[npx, npy]
    zdata = clf.predict(sample_points)
    figure(figsize=(8, 6))
    zdata = zdata.reshape(xxx.shape)
    contourf(xxx, yyy, zdata, cmap=cm.Paired, alpha=0.8)
    scatter(xdata[:, 0], xdata[:, 1], c=ydata.astype(npfloat))


def wrapper():
    """
    wrapper function
    """
    xdata, ydata = create_clustered_data(100, 5)
    figure(figsize=(8, 6))
    scatter(xdata[:, 0], xdata[:, 1], c=ydata.astype(npfloat))
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(xdata)
    xdata = scaling.transform(xdata)
    figure(figsize=(8, 6))
    scatter(xdata[:, 0], xdata[:, 1], c=ydata.astype(npfloat))
    cnum = 1.0
    svc = SVC(kernel="linear", C=cnum).fit(xdata, ydata)
    plot_predictions(svc, xdata, ydata)
    print(svc.predict(scaling.transform([[200000, 40]])))
    print(svc.predict(scaling.transform([[50000, 65]])))


if __name__ == "__main__":
    wrapper()
    show()
