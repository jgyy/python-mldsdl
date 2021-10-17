"""
K-Fold Cross Validation
"""
from types import SimpleNamespace
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import load_iris
from sklearn.svm import SVC


def wrapper():
    """
    wrapper function
    """
    iris = SimpleNamespace(**load_iris())
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.4, random_state=0
    )
    clf = SVC(kernel="linear", C=1).fit(x_train, y_train)
    clf.score(x_test, y_test)
    scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    print(scores)
    print(scores.mean())
    clf = SVC(kernel="poly", C=1)
    scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    print(scores)
    print(scores.mean())
    clf = SVC(kernel="poly", C=1).fit(x_train, y_train)
    print(clf.score(x_test, y_test))


if __name__ == "__main__":
    wrapper()
